"""
Training script for IV Surface Diffusion Model
Includes data preprocessing, EWMA calculation, and training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Dict, List
from scipy import ndimage
import pandas as pd
from tqdm import tqdm


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

class IVDataPreprocessor:
    """Preprocesses implied volatility surfaces following the paper's methodology."""
    
    def __init__(
        self,
        moneyness_grid: np.ndarray = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
        tenor_grid: np.ndarray = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1.0]),
        h1: float = 0.002,  # bandwidth for moneyness
        h2: float = 0.046,  # bandwidth for tenor
    ):
        self.moneyness_grid = moneyness_grid
        self.tenor_grid = tenor_grid
        self.h1 = h1
        self.h2 = h2
        self.mean_surface = None
        self.std_surface = None
    
    def gaussian_kernel_2d(self, x: float, y: float) -> float:
        """2D Gaussian kernel for smoothing."""
        return (1 / (2 * np.pi)) * np.exp(-x**2 / (2 * self.h1) - y**2 / (2 * self.h2))
    
    def vega_weighted_smoothing(
        self,
        iv_data: np.ndarray,
        moneyness: np.ndarray,
        tenors: np.ndarray,
        vegas: np.ndarray
    ) -> np.ndarray:
        """
        Apply Vega-weighted Nadaraya-Watson kernel smoothing.
        
        Args:
            iv_data: Raw IV observations [N]
            moneyness: Moneyness values [N]
            tenors: Tenor values [N]
            vegas: Vega values [N]
        
        Returns:
            Smoothed IV surface [N_m, N_tau]
        """
        N_m, N_tau = len(self.moneyness_grid), len(self.tenor_grid)
        smoothed_surface = np.zeros((N_m, N_tau))
        
        for i, m_target in enumerate(self.moneyness_grid):
            for j, tau_target in enumerate(self.tenor_grid):
                numerator = 0.0
                denominator = 0.0
                
                for k in range(len(iv_data)):
                    kernel_val = self.gaussian_kernel_2d(
                        moneyness[k] - m_target,
                        tenors[k] - tau_target
                    )
                    weight = vegas[k] * kernel_val
                    numerator += weight * iv_data[k]
                    denominator += weight
                
                if denominator > 0:
                    smoothed_surface[i, j] = numerator / denominator
                else:
                    # Fallback: use nearest neighbor
                    smoothed_surface[i, j] = iv_data[0]  # Placeholder
        
        return smoothed_surface
    
    def log_transform(self, iv_surface: np.ndarray) -> np.ndarray:
        """Transform IV to log space."""
        return np.log(iv_surface + 1e-8)
    
    def normalize(self, log_iv_surfaces: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize log-IV surfaces using per-grid-point statistics.
        
        Args:
            log_iv_surfaces: Log-transformed IV surfaces [T, N_m, N_tau]
            fit: If True, compute and store mean/std from this data
        
        Returns:
            Normalized surfaces [T, N_m, N_tau]
        """
        if fit:
            self.mean_surface = np.mean(log_iv_surfaces, axis=0)
            self.std_surface = np.std(log_iv_surfaces, axis=0) + 1e-8
        
        normalized = (log_iv_surfaces - self.mean_surface) / self.std_surface
        return normalized
    
    def inverse_transform(self, normalized_log_iv: np.ndarray) -> np.ndarray:
        """Transform from normalized log-space back to IV."""
        log_iv = normalized_log_iv * self.std_surface + self.mean_surface
        iv = np.exp(log_iv)
        return iv
    
    def process_dataset(
        self,
        raw_surfaces: np.ndarray,
        fit_normalization: bool = True
    ) -> np.ndarray:
        """
        Complete preprocessing pipeline.
        
        Args:
            raw_surfaces: Raw IV surfaces [T, N_m, N_tau]
            fit_normalization: Whether to fit normalization parameters
        
        Returns:
            Processed surfaces [T, N_m, N_tau]
        """
        # Log transform
        log_surfaces = self.log_transform(raw_surfaces)
        
        # Normalize
        normalized = self.normalize(log_surfaces, fit=fit_normalization)
        
        return normalized


# ============================================================================
# EWMA CALCULATION
# ============================================================================

def calculate_ewma(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Calculate Exponentially Weighted Moving Average.
    
    Args:
        data: Time series data [T, ...]
        alpha: Smoothing factor
    
    Returns:
        EWMA values [T, ...]
    """
    T = len(data)
    ewma = np.zeros_like(data)
    ewma[0] = data[0]
    
    for t in range(1, T):
        ewma[t] = alpha * data[t] + (1 - alpha) * ewma[t-1]
    
    return ewma


def calculate_conditioning_variables(
    returns: np.ndarray,
    vix_returns: np.ndarray,
    alpha_trend_short: float = 0.156,
    alpha_trend_long: float = 0.118,
    alpha_vol_short: float = 0.3,
    alpha_vol_long: float = 0.15,
) -> np.ndarray:
    """
    Calculate the 5 scalar conditioning variables.
    
    Args:
        returns: Daily returns of underlying [T]
        vix_returns: Daily VIX returns [T]
        alpha_*: EWMA smoothing factors
    
    Returns:
        Conditioning matrix [T, 5]
    """
    T = len(returns)
    cond_vars = np.zeros((T, 5))
    
    # Calculate EWMAs
    cond_vars[:, 0] = calculate_ewma(returns, alpha_trend_short)
    cond_vars[:, 1] = calculate_ewma(returns, alpha_trend_long)
    cond_vars[:, 2] = calculate_ewma(returns**2, alpha_vol_short)
    cond_vars[:, 3] = calculate_ewma(returns**2, alpha_vol_long)
    cond_vars[:, 4] = vix_returns
    
    return cond_vars


# ============================================================================
# DATASET
# ============================================================================

class IVSurfaceDataset(Dataset):
    """Dataset for conditional IV surface forecasting."""
    
    def __init__(
        self,
        surfaces: np.ndarray,  # [T, N_m, N_tau]
        scalar_cond: np.ndarray,  # [T, 5]
        ewma_short_span: int = 5,
        ewma_long_span: int = 20,
    ):
        """
        Args:
            surfaces: Normalized log-IV surfaces
            scalar_cond: Scalar conditioning variables
            ewma_short_span: Days for short-term surface EWMA
            ewma_long_span: Days for long-term surface EWMA
        """
        self.surfaces = surfaces
        self.scalar_cond = scalar_cond
        
        # Calculate surface EWMAs
        self.ewma_short = self._calculate_surface_ewma(surfaces, ewma_short_span)
        self.ewma_long = self._calculate_surface_ewma(surfaces, ewma_long_span)
        
        # Valid indices (need history for EWMAs and target is next day)
        self.valid_start = max(ewma_short_span, ewma_long_span)
        self.valid_end = len(surfaces) - 1
    
    def _calculate_surface_ewma(self, surfaces: np.ndarray, span: int) -> np.ndarray:
        """Calculate EWMA for entire surfaces."""
        alpha = 2 / (span + 1)
        T, H, W = surfaces.shape
        ewma = np.zeros_like(surfaces)
        ewma[0] = surfaces[0]
        
        for t in range(1, T):
            ewma[t] = alpha * surfaces[t] + (1 - alpha) * ewma[t-1]
        
        return ewma
    
    def __len__(self) -> int:
        return self.valid_end - self.valid_start
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
                - x_0: Target surface (next day) [1, H, W]
                - x_input: 4-channel input [4, H, W]
                - scalar_cond: Scalar conditioning [5]
        """
        actual_idx = idx + self.valid_start
        
        # Target (next day)
        x_0 = self.surfaces[actual_idx + 1]
        
        # 4-channel input:
        # - Current day surface
        # - Short-term EWMA
        # - Long-term EWMA  
        # - Placeholder for noisy target (will be filled during training)
        current_surface = self.surfaces[actual_idx]
        ewma_s = self.ewma_short[actual_idx]
        ewma_l = self.ewma_long[actual_idx]
        
        x_input = np.stack([current_surface, ewma_s, ewma_l, np.zeros_like(current_surface)], axis=0)
        
        # Scalar conditioning
        scalar_cond = self.scalar_cond[actual_idx]
        
        return {
            'x_0': torch.FloatTensor(x_0[None, :, :]),  # [1, H, W]
            'x_input': torch.FloatTensor(x_input),  # [4, H, W]
            'scalar_cond': torch.FloatTensor(scalar_cond),  # [5]
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(
    model: nn.Module,
    ddpm: 'ConditionalDDPM',
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    ema: 'EMA',
    mean_surface: torch.Tensor,
    std_surface: torch.Tensor,
    device: str,
    grad_clip: float = 0.15,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {'total': 0, 'mse': 0, 'arb': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        x_0 = batch['x_0'].to(device)
        x_input = batch['x_input'].to(device)
        scalar_cond = batch['scalar_cond'].to(device)
        
        # Compute loss
        loss, loss_dict = ddpm.compute_loss(
            x_0, x_input, scalar_cond,
            mean_surface, std_surface
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        # Accumulate losses
        for key in total_losses:
            total_losses[key] += loss_dict[key]
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'mse': f"{loss_dict['mse']:.4f}",
            'arb': f"{loss_dict['arb']:.4f}"
        })
    
    # Average losses
    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in total_losses.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    ddpm: 'ConditionalDDPM',
    dataloader: DataLoader,
    mean_surface: torch.Tensor,
    std_surface: torch.Tensor,
    device: str,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_losses = {'total': 0, 'mse': 0, 'arb': 0}
    
    for batch in tqdm(dataloader, desc="Validation"):
        x_0 = batch['x_0'].to(device)
        x_input = batch['x_input'].to(device)
        scalar_cond = batch['scalar_cond'].to(device)
        
        _, loss_dict = ddpm.compute_loss(
            x_0, x_input, scalar_cond,
            mean_surface, std_surface
        )
        
        for key in total_losses:
            total_losses[key] += loss_dict[key]
    
    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in total_losses.items()}


def train_model(
    model: nn.Module,
    ddpm: 'ConditionalDDPM',
    train_loader: DataLoader,
    val_loader: DataLoader,
    mean_surface: torch.Tensor,
    std_surface: torch.Tensor,
    device: str,
    num_epochs: int = 2000,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    ema_decay: float = 0.995,
    patience: int = 300,
    lr_factor: float = 0.8,
    min_lr: float = 1e-6,
):
    """
    Complete training procedure with early stopping and LR scheduling.
    
    Args:
        model: U-Net model
        ddpm: Diffusion model wrapper
        train_loader: Training data loader
        val_loader: Validation data loader
        mean_surface: Mean surface for inverse transform
        std_surface: Std surface for inverse transform
        device: Device to train on
        num_epochs: Maximum number of epochs
        lr: Initial learning rate
        weight_decay: AdamW weight decay
        ema_decay: EMA decay rate
        patience: Patience for LR scheduler and early stopping
        lr_factor: Factor to reduce LR
        min_lr: Minimum learning rate
    """
    from iv_diffusion_main import EMA  # Import EMA class
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=patience, min_lr=min_lr
    )
    
    # EMA
    ema = EMA(model, decay=ema_decay)
    
    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Train
        train_losses = train_epoch(
            model, ddpm, train_loader, optimizer, ema,
            mean_surface, std_surface, device
        )
        
        print(f"Train - Total: {train_losses['total']:.4f}, "
              f"MSE: {train_losses['mse']:.4f}, "
              f"Arb: {train_losses['arb']:.4f}")
        
        # Validate with EMA model
        ema.apply_shadow()
        val_losses = validate(
            model, ddpm, val_loader,
            mean_surface, std_surface, device
        )
        ema.restore()
        
        print(f"Val - Total: {val_losses['total']:.4f}, "
              f"MSE: {val_losses['mse']:.4f}, "
              f"Arb: {val_losses['arb']:.4f}")
        
        # LR scheduling
        scheduler.step(val_losses['total'])
        
        # Early stopping check
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            epochs_without_improvement = 0
            
            # Save best model
            ema.apply_shadow()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
            }, 'best_model.pt')
            ema.restore()
            print("✓ Saved best model")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience * 2:
            print(f"\nEarly stopping after {patience * 2} epochs without improvement")
            break
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Training script loaded successfully!")
    print("\nTo train the model:")
    print("1. Prepare your IV surface data")
    print("2. Calculate returns and VIX data")
    print("3. Create train/val datasets")
    print("4. Call train_model() with your data")
    
    # Example synthetic data generation
    print("\n" + "="*60)
    print("Example: Creating synthetic data for testing")
    print("="*60)
    
    # Synthetic data dimensions
    T = 1000  # days
    N_m, N_tau = 9, 9
    
    # Generate synthetic IV surfaces (for demonstration only)
    np.random.seed(42)
    raw_surfaces = 0.2 + 0.05 * np.random.randn(T, N_m, N_tau)
    raw_surfaces = np.clip(raw_surfaces, 0.05, 0.8)
    
    # Generate synthetic returns
    returns = 0.001 * np.random.randn(T)
    vix_returns = 0.002 * np.random.randn(T)
    
    # Preprocess
    preprocessor = IVDataPreprocessor()
    normalized_surfaces = preprocessor.process_dataset(raw_surfaces, fit_normalization=True)
    
    # Calculate conditioning variables
    scalar_cond = calculate_conditioning_variables(returns, vix_returns)
    
    # Standardize scalar conditioning (fit on train data)
    scalar_mean = scalar_cond[:800].mean(axis=0)
    scalar_std = scalar_cond[:800].std(axis=0) + 1e-8
    scalar_cond_normalized = (scalar_cond - scalar_mean) / scalar_std
    
    # Create datasets
    train_dataset = IVSurfaceDataset(normalized_surfaces[:800], scalar_cond_normalized[:800])
    val_dataset = IVSurfaceDataset(normalized_surfaces[800:900], scalar_cond_normalized[800:900])
    
    print(f"\nDataset created:")
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    print(f"- Surface shape: {N_m}x{N_tau}")
    print(f"- Mean surface range: [{preprocessor.mean_surface.min():.3f}, {preprocessor.mean_surface.max():.3f}]")
    print(f"- Std surface range: [{preprocessor.std_surface.min():.3f}, {preprocessor.std_surface.max():.3f}]")
