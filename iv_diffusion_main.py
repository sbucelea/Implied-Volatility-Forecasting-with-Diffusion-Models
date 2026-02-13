"""
Implied Volatility Surface Forecasting with Conditional DDPM
Based on: Jin & Agarwal (2025) - "Forecasting implied volatility surface with generative diffusion models"

Main implementation file with all core components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


# ============================================================================
# DIFFUSION SCHEDULE
# ============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for beta_t as proposed in improved DDPM paper.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset to prevent beta_t from being too small near t=0
    
    Returns:
        beta_t schedule of shape (timesteps,)
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionSchedule:
    """Manages the variance schedule for the diffusion process."""
    
    def __init__(self, num_timesteps: int = 500, schedule_type: str = 'cosine'):
        self.num_timesteps = num_timesteps
        
        if schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Calculations for SNR weighting
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """Calculate Signal-to-Noise Ratio for given timesteps."""
        alphas_cumprod_t = self.alphas_cumprod[t]
        return alphas_cumprod_t / (1.0 - alphas_cumprod_t + 1e-8)


# ============================================================================
# U-NET ARCHITECTURE WITH FiLM CONDITIONING
# ============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""
    
    def __init__(self, emb_dim: int, num_channels: int, hidden_dim: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_channels * 2)  # gamma and beta
        )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
            emb: Conditioning embedding [B, emb_dim]
        
        Returns:
            Modulated features [B, C, H, W]
        """
        emb = self.mlp(emb)
        gamma, beta = emb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]  # [B, C, 1, 1]
        beta = beta[:, :, None, None]    # [B, C, 1, 1]
        return gamma * x + beta


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        
        # Print to verify initialization (temporary debug)
        print(f"ConvBlock init: in={in_channels}, out={out_channels}, emb_dim={emb_dim}")
        
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0:
            num_groups -= 1
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU()
        )
        
        # NO hardcoded 128 or 256 here!
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, out_channels * 2),
            nn.SiLU(),
            nn.Linear(out_channels * 2, out_channels)
        )
    
    def forward(self, x, emb):
        x = self.conv(x)
        emb = self.mlp(emb)
        emb = emb[:, :, None, None]
        return x + emb


class UNet(nn.Module):
    """
    U-Net architecture for noise prediction in diffusion model.
    Adapted for 9x9 IV surfaces with conditional generation.
    """
    
    def __init__(
    self,
    in_channels: int = 4,
    out_channels: int = 1,
    enc_channels: int = 16,
    bottle_channels: int = 32,
    time_emb_dim: int = 14,
    scalar_cond_dim: int = 5,
    emb_dim: int = 24,
    ):
        super().__init__()
    
        print(f"UNet init - enc_channels: {enc_channels}, emb_dim: {emb_dim}")
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(10),
            nn.Linear(10, time_emb_dim),
            nn.SiLU(),
        )
        
        # Scalar conditioning MLP
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_cond_dim, 10),
            nn.SiLU(),
        )
        
        # Total embedding dimension
        self.emb_dim = emb_dim
        
        # Encoder
        print(f"Creating enc1: in={in_channels}, out={enc_channels}, emb={emb_dim}")
        self.enc1 = ConvBlock(in_channels, enc_channels, emb_dim)
    
        print(f"Creating enc2: in={enc_channels}, out={enc_channels * 2}, emb={emb_dim}")
        self.enc2 = ConvBlock(enc_channels, enc_channels * 2, emb_dim)
        
        # Bottleneck
        self.bottle = ConvBlock(enc_channels * 2, bottle_channels, emb_dim)
        
        # Decoder
        self.dec2 = ConvBlock(bottle_channels + enc_channels * 2, enc_channels * 2, emb_dim)
        self.dec1 = ConvBlock(enc_channels * 2 + enc_channels, enc_channels, emb_dim)
        
        # Output
        self.out = nn.Conv2d(enc_channels, out_channels, kernel_size=1)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        scalar_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input [B, 4, 9, 9]
            t: Timestep [B]
            scalar_cond: Scalar conditioning [B, 5]
        
        Returns:
            Predicted noise [B, 1, 9, 9]
        """
        # Create combined embedding
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]
        s_emb = self.scalar_mlp(scalar_cond)  # [B, 10]
        emb = torch.cat([t_emb, s_emb], dim=1)  # [B, emb_dim]
        
        # Encoder
        e1 = self.enc1(x, emb)  # [B, 16, 9, 9]
        e2 = self.enc2(e1, emb)  # [B, 32, 9, 9]
        
        # Bottleneck
        b = self.bottle(e2, emb)  # [B, 30, 9, 9]
        
        # Decoder with skip connections
        d2 = self.dec2(torch.cat([b, e2], dim=1), emb)  # [B, 32, 9, 9]
        d1 = self.dec1(torch.cat([d2, e1], dim=1), emb)  # [B, 16, 9, 9]
        
        # Output
        out = self.out(d1)  # [B, 1, 9, 9]
        return out


# ============================================================================
# ARBITRAGE PENALTY CALCULATION
# ============================================================================

def black_scholes_call(S: float, K: torch.Tensor, tau: torch.Tensor, 
                       sigma: torch.Tensor, r: float = 0.0) -> torch.Tensor:
    """
    Calculate Black-Scholes call price (normalized by S).
    
    Args:
        S: Spot price (scalar)
        K: Strike prices [B, N_m, N_tau]
        tau: Time to maturity [B, N_m, N_tau]
        sigma: Implied volatilities [B, N_m, N_tau]
        r: Risk-free rate
    
    Returns:
        Normalized call prices [B, N_m, N_tau]
    """
    m = K / S  # Moneyness
    d1 = (torch.log(1/m) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau) + 1e-8)
    d2 = d1 - sigma * torch.sqrt(tau)
    
    from torch.distributions import Normal
    norm = Normal(0, 1)
    
    call_price = norm.cdf(d1) - m * torch.exp(-r * tau) * norm.cdf(d2)
    return call_price


def calculate_arbitrage_penalty(
    sigma: torch.Tensor,
    moneyness_grid: torch.Tensor,
    tenor_grid: torch.Tensor,
    S: float = 100.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate three types of arbitrage penalties.
    
    Args:
        sigma: IV surface [B, N_m, N_tau]
        moneyness_grid: Moneyness values [N_m]
        tenor_grid: Tenor values [N_tau]
        S: Spot price
    
    Returns:
        Tuple of (p1_calendar, p2_call_spread, p3_butterfly)
    """
    B, N_m, N_tau = sigma.shape
    device = sigma.device
    
    # Expand grids
    m = moneyness_grid[None, :, None].expand(B, N_m, N_tau).to(device)
    tau = tenor_grid[None, None, :].expand(B, N_m, N_tau).to(device)
    K = m * S
    
    # Calculate call prices
    c = black_scholes_call(S, K, tau, sigma)
    
    # P1: Calendar spread arbitrage (non-decreasing in tenor)
    dc_dtau = c[:, :, 1:] - c[:, :, :-1]  # [B, N_m, N_tau-1]
    dtau = tenor_grid[1:] - tenor_grid[:-1]  # [N_tau-1]
    dtau = dtau[None, None, :].to(device)
    p1 = F.relu(-dc_dtau / (dtau + 1e-8)).sum(dim=[1, 2])
    
    # P2: Call spread arbitrage (negative slope in strike)
    dc_dm = c[:, 1:, :] - c[:, :-1, :]  # [B, N_m-1, N_tau]
    dm = moneyness_grid[1:] - moneyness_grid[:-1]  # [N_m-1]
    dm = dm[None, :, None].to(device)
    p2 = F.relu(dc_dm / (dm + 1e-8)).sum(dim=[1, 2])
    
    # P3: Butterfly arbitrage (convexity in strike)
    if N_m >= 3:
        d2c_dm2 = (c[:, 2:, :] - c[:, 1:-1, :]) / (dm[:, 1:, :] + 1e-8) - \
                  (c[:, 1:-1, :] - c[:, :-2, :]) / (dm[:, :-1, :] + 1e-8)
        p3 = F.relu(-d2c_dm2).sum(dim=[1, 2])
    else:
        p3 = torch.zeros(B, device=device)
    
    return p1, p2, p3


# ============================================================================
# DIFFUSION MODEL (Training & Sampling)
# ============================================================================

class ConditionalDDPM:
    """Conditional Denoising Diffusion Probabilistic Model for IV surfaces."""
    
    def __init__(
        self,
        model: nn.Module,
        schedule: DiffusionSchedule,
        lambda_arb: float = 0.01,
        moneyness_grid: Optional[torch.Tensor] = None,
        tenor_grid: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.schedule = schedule
        self.lambda_arb = lambda_arb
        
        # Default grids from paper
        if moneyness_grid is None:
            moneyness_grid = torch.tensor([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        if tenor_grid is None:
            tenor_grid = torch.tensor([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1.0])
        
        self.moneyness_grid = moneyness_grid
        self.tenor_grid = tenor_grid
    
    def forward_diffusion(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: q(x_t | x_0).
        
        Args:
            x_0: Clean data [B, 1, 9, 9]
            t: Timesteps [B]
            noise: Optional pre-sampled noise
        
        Returns:
            Tuple of (x_t, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise
    
    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor
    ) -> torch.Tensor:
        """Denoise to get estimate of x_0."""
        sqrt_alphas_cumprod_t = self.schedule.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8)
        return x_0_pred
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        x_input: torch.Tensor,
        scalar_cond: torch.Tensor,
        mean_surface: torch.Tensor,
        std_surface: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss with SNR-weighted arbitrage penalty.
        
        Args:
            x_0: Target surface in normalized log-space [B, 1, 9, 9]
            x_input: 4-channel input [B, 4, 9, 9]
            scalar_cond: Scalar conditioning [B, 5]
            mean_surface: Mean for denormalization [9, 9]
            std_surface: Std for denormalization [9, 9]
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(0, self.schedule.num_timesteps, (B,), device=device).long()
        
        # Forward diffusion
        x_t, noise = self.forward_diffusion(x_0, t)
        
        # Replace the noisy target channel in x_input
        x_input_with_noise = x_input.clone()
        x_input_with_noise[:, -1:, :, :] = x_t
        
        # Predict noise
        noise_pred = self.model(x_input_with_noise, t, scalar_cond)
        
        # MSE loss
        mse_loss = F.mse_loss(noise_pred, noise)
        
        # Arbitrage penalty with SNR weighting
        if self.lambda_arb > 0:
            # Get x_0 estimate
            x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
            
            # Clip for stability
            x_0_pred = torch.clamp(x_0_pred, -3, 3)
            
            # Inverse transform to IV space
            sigma_pred = self.inverse_transform(x_0_pred, mean_surface, std_surface)
            
            # Calculate arbitrage penalties
            p1, p2, p3 = calculate_arbitrage_penalty(
                sigma_pred.squeeze(1),
                self.moneyness_grid,
                self.tenor_grid
            )
            
            arb_penalty = p1 + p2 + p3
            
            # SNR weighting
            snr = self.schedule.get_snr(t)
            weighted_arb = (snr * arb_penalty).mean()
            
            total_loss = mse_loss + self.lambda_arb * weighted_arb
        else:
            weighted_arb = torch.tensor(0.0, device=device)
            total_loss = mse_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'arb': weighted_arb.item() if isinstance(weighted_arb, torch.Tensor) else weighted_arb,
        }
        
        return total_loss, loss_dict
    
    @staticmethod
    def inverse_transform(
        x_normalized: torch.Tensor,
        mean_surface: torch.Tensor,
        std_surface: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform from normalized log-space back to IV.
        
        Args:
            x_normalized: Normalized log-IV [B, 1, 9, 9]
            mean_surface: Mean [9, 9]
            std_surface: Std [9, 9]
        
        Returns:
            IV surface [B, 9, 9]
        """
        # Denormalize
        x_log = x_normalized.squeeze(1) * std_surface[None, :, :] + mean_surface[None, :, :]
        # Exponentiate
        sigma = torch.exp(x_log)
        return sigma
    
    @torch.no_grad()
    def sample(
        self,
        x_input: torch.Tensor,
        scalar_cond: torch.Tensor,
        num_samples: int = 1,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the reverse diffusion process.
        
        Args:
            x_input: 4-channel input (first 3 channels) [B, 4, 9, 9]
            scalar_cond: Scalar conditioning [B, 5]
            num_samples: Number of samples to generate per condition
            clip_denoised: Whether to clip the denoised prediction
        
        Returns:
            Generated samples [B*num_samples, 1, 9, 9]
        """
        device = next(self.model.parameters()).device
        B = x_input.shape[0]
        
        # Replicate inputs for multiple samples
        x_input = x_input.repeat(num_samples, 1, 1, 1)
        scalar_cond = scalar_cond.repeat(num_samples, 1)
        
        # Start from pure noise
        x_t = torch.randn(B * num_samples, 1, 9, 9, device=device)
        
        # Reverse diffusion
        for i in reversed(range(self.schedule.num_timesteps)):
            t = torch.full((B * num_samples,), i, device=device, dtype=torch.long)
            
            # Update noisy channel
            x_input_t = x_input.clone()
            x_input_t[:, -1:, :, :] = x_t
            
            # Predict noise
            noise_pred = self.model(x_input_t, t, scalar_cond)
            
            # Compute mean of reverse distribution
            sqrt_recip_alphas_t = self.schedule.sqrt_recip_alphas[t][:, None, None, None]
            betas_t = self.schedule.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
            
            model_mean = sqrt_recip_alphas_t * (
                x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
            )
            
            if clip_denoised:
                model_mean = torch.clamp(model_mean, -3, 3)
            
            if i > 0:
                noise = torch.randn_like(x_t)
                posterior_variance_t = self.schedule.posterior_variance[t][:, None, None, None]
                x_t = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x_t = model_mean
        
        return x_t


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================================

class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


if __name__ == "__main__":
    print("IV Surface Diffusion Model - Implementation Complete")
    print("\nKey Components:")
    print("- Cosine beta schedule for diffusion")
    print("- U-Net with FiLM conditioning")
    print("- SNR-weighted arbitrage penalty")
    print("- EMA for stable training")
    print("\nNext steps: Implement data preprocessing and training loop")
