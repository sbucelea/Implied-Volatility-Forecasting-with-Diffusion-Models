"""
Complete End-to-End Example
Demonstrates the full pipeline from data preparation to evaluation
"""
import sys
import os

# Clear any cached modules
for module in list(sys.modules.keys()):
    if 'iv_diffusion' in module:
        del sys.modules[module]

# Now import fresh
from iv_diffusion_main import *
from iv_diffusion_train import *


import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from iv_diffusion_main import (
    DiffusionSchedule, UNet, ConditionalDDPM, EMA
)
from iv_diffusion_train import (
    IVDataPreprocessor, calculate_conditioning_variables,
    IVSurfaceDataset, train_model
)
from iv_diffusion_eval import (
    generate_evaluation_report, plot_surface_comparison,
    plot_time_series_slice, plot_arbitrage_over_time, plot_distribution_comparison
)


# ============================================================================
# STEP 1: DATA PREPARATION
# ============================================================================

def prepare_synthetic_data(
    num_days: int = 1500,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42
):
    """
    Prepare synthetic data for demonstration.
    In practice, replace this with real OptionMetrics data.
    
    Returns:
        Dictionary with all prepared data
    """
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Grid configuration from paper
    moneyness_grid = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    tenor_grid = np.array([1/252, 1/52, 2/52, 1/12, 1/6, 1/4, 1/2, 3/4, 1.0])
    N_m, N_tau = len(moneyness_grid), len(tenor_grid)
    
    print(f"\nGrid configuration:")
    print(f"- Moneyness points: {N_m}")
    print(f"- Tenor points: {N_tau}")
    print(f"- Total grid points: {N_m * N_tau}")
    
    # Generate synthetic IV surfaces with realistic characteristics
    # Surfaces should have: smile effect, term structure, temporal evolution
    print(f"\nGenerating {num_days} days of synthetic IV data...")
    
    surfaces = []
    base_vol = 0.20
    
    for t in range(num_days):
        surface = np.zeros((N_m, N_tau))
        
        # Time-varying component (market regime)
        regime = 0.05 * np.sin(2 * np.pi * t / 252) + 0.02 * np.random.randn()
        
        for i, m in enumerate(moneyness_grid):
            for j, tau in enumerate(tenor_grid):
                # Volatility smile (higher for OTM puts and calls)
                smile = 0.1 * (m - 1.0)**2
                
                # Term structure (slightly increasing)
                term = 0.03 * np.sqrt(tau)
                
                # Add noise
                noise = 0.01 * np.random.randn()
                
                surface[i, j] = base_vol + smile + term + regime + noise
        
        surfaces.append(surface)
    
    surfaces = np.array(surfaces)
    surfaces = np.clip(surfaces, 0.05, 0.8)  # Realistic bounds
    
    # Generate underlying returns and VIX
    print("Generating returns and VIX data...")
    returns = 0.0005 + 0.015 * np.random.randn(num_days)
    vix_returns = 0.001 + 0.02 * np.random.randn(num_days)
    
    # Preprocess surfaces
    print("\nPreprocessing surfaces...")
    preprocessor = IVDataPreprocessor(moneyness_grid, tenor_grid)
    
    # Split data
    train_end = int(num_days * train_split)
    val_end = int(num_days * (train_split + val_split))
    
    # Fit normalization on training data only
    train_surfaces_normalized = preprocessor.process_dataset(
        surfaces[:train_end], fit_normalization=True
    )
    val_surfaces_normalized = preprocessor.process_dataset(
        surfaces[train_end:val_end], fit_normalization=False
    )
    test_surfaces_normalized = preprocessor.process_dataset(
        surfaces[val_end:], fit_normalization=False
    )
    
    # Combine for full dataset
    all_surfaces_normalized = np.concatenate([
        train_surfaces_normalized,
        val_surfaces_normalized,
        test_surfaces_normalized
    ])
    
    print(f"Normalization statistics:")
    print(f"- Mean surface range: [{preprocessor.mean_surface.min():.3f}, {preprocessor.mean_surface.max():.3f}]")
    print(f"- Std surface range: [{preprocessor.std_surface.min():.3f}, {preprocessor.std_surface.max():.3f}]")
    
    # Calculate conditioning variables
    print("\nCalculating conditioning variables...")
    scalar_cond = calculate_conditioning_variables(returns, vix_returns)
    
    # Normalize scalar conditioning (fit on training data)
    scalar_mean = scalar_cond[:train_end].mean(axis=0)
    scalar_std = scalar_cond[:train_end].std(axis=0) + 1e-8
    scalar_cond_norm = (scalar_cond - scalar_mean) / scalar_std
    
    print("\nData splits:")
    print(f"- Training: {train_end} days")
    print(f"- Validation: {val_end - train_end} days")
    print(f"- Test: {num_days - val_end} days")
    
    return {
        'surfaces_raw': surfaces,
        'surfaces_normalized': all_surfaces_normalized,
        'scalar_cond': scalar_cond_norm,
        'preprocessor': preprocessor,
        'scalar_mean': scalar_mean,
        'scalar_std': scalar_std,
        'moneyness_grid': moneyness_grid,
        'tenor_grid': tenor_grid,
        'split_indices': {
            'train_end': train_end,
            'val_end': val_end,
            'test_end': num_days
        }
    }


# ============================================================================
# STEP 2: MODEL SETUP
# ============================================================================

def setup_model(device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Setup the diffusion model architecture.
    
    Returns:
        Tuple of (model, schedule, ddpm)
    """
    print("\n" + "=" * 60)
    print("STEP 2: MODEL SETUP")
    print("=" * 60)
    
    print(f"\nDevice: {device}")
    
    # Diffusion schedule
    print("\nCreating diffusion schedule...")
    schedule = DiffusionSchedule(num_timesteps=500, schedule_type='cosine')
    print(f"- Number of timesteps: {schedule.num_timesteps}")
    print(f"- Beta range: [{schedule.betas.min():.6f}, {schedule.betas.max():.6f}]")
    
    # U-Net model
    print("\nCreating U-Net model...")
    model = UNet(
        in_channels=4,
        out_channels=1,
        enc_channels=32,  
        bottle_channels=32,
        time_emb_dim=14,
        scalar_cond_dim=5,
        emb_dim=24
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    
    # DDPM wrapper
    print("\nCreating DDPM wrapper...")
    ddpm = ConditionalDDPM(
        model=model,
        schedule=schedule,
        lambda_arb=0.01
    )
    print(f"- Arbitrage penalty weight: {ddpm.lambda_arb}")
    
    return model, schedule, ddpm


# ============================================================================
# STEP 3: TRAINING
# ============================================================================

def train_diffusion_model(
    model,
    ddpm,
    data_dict,
    device,
    num_epochs: int = 100,  # Reduced for demo
    batch_size: int = 64,
):
    """
    Train the diffusion model.
    """
    print("\n" + "=" * 60)
    print("STEP 3: TRAINING")
    print("=" * 60)
    
    # Extract data
    surfaces = data_dict['surfaces_normalized']
    scalar_cond = data_dict['scalar_cond']
    split = data_dict['split_indices']
    preprocessor = data_dict['preprocessor']
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = IVSurfaceDataset(
        surfaces[:split['train_end']],
        scalar_cond[:split['train_end']]
    )
    val_dataset = IVSurfaceDataset(
        surfaces[split['train_end']:split['val_end']],
        scalar_cond[split['train_end']:split['val_end']]
    )
    
    print(f"- Training samples: {len(train_dataset)}")
    print(f"- Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Convert normalization surfaces to torch
    mean_surface = torch.FloatTensor(preprocessor.mean_surface).to(device)
    std_surface = torch.FloatTensor(preprocessor.std_surface).to(device)
    
    # Train
    print(f"\nStarting training for {num_epochs} epochs...")
    train_model(
        model=model,
        ddpm=ddpm,
        train_loader=train_loader,
        val_loader=val_loader,
        mean_surface=mean_surface,
        std_surface=std_surface,
        device=device,
        num_epochs=num_epochs,
        lr=3e-4,
        patience=30
    )
    
    print("\nTraining complete!")


# ============================================================================
# STEP 4: SAMPLING AND EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_model(
    model,
    ddpm,
    data_dict,
    device,
    num_samples: int = 10
):
    """
    Generate samples and evaluate the model.
    """
    print("\n" + "=" * 60)
    print("STEP 4: SAMPLING AND EVALUATION")
    print("=" * 60)
    
    model.eval()
    
    # Load best model
    print("\nLoading best model...")
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Prepare test data
    surfaces = data_dict['surfaces_normalized']
    scalar_cond = data_dict['scalar_cond']
    split = data_dict['split_indices']
    preprocessor = data_dict['preprocessor']
    
    test_dataset = IVSurfaceDataset(
        surfaces[split['val_end']:],
        scalar_cond[split['val_end']:]
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Generate samples
    print(f"\nGenerating {num_samples} samples for each test day...")
    all_samples = []
    all_targets = []
    
    from tqdm import tqdm
    for idx in tqdm(range(len(test_dataset)), desc="Sampling"):
        batch = test_dataset[idx]
        
        # Prepare inputs
        x_input = batch['x_input'].unsqueeze(0).to(device)  # [1, 4, 9, 9]
        scalar_cond_batch = batch['scalar_cond'].unsqueeze(0).to(device)  # [1, 5]
        x_0_target = batch['x_0'].cpu().numpy()  # [1, 9, 9]
        
        # Generate samples
        samples = ddpm.sample(
            x_input=x_input,
            scalar_cond=scalar_cond_batch,
            num_samples=num_samples,
            clip_denoised=True
        )  # [num_samples, 1, 9, 9]
        
        samples = samples.cpu().numpy()
        all_samples.append(samples)
        all_targets.append(x_0_target)
    
    # Convert to arrays
    all_samples = np.array(all_samples)  # [T, N_samples, 1, 9, 9]
    all_targets = np.array(all_targets)  # [T, 1, 9, 9]
    
    # Reshape
    all_samples = all_samples.squeeze(2)  # [T, N_samples, 9, 9]
    all_targets = all_targets.squeeze(1)  # [T, 9, 9]
    
    # Inverse transform to IV space
    print("\nInverse transforming to IV space...")
    mean_surf = preprocessor.mean_surface
    std_surf = preprocessor.std_surface
    
    samples_iv = []
    for t in range(len(all_samples)):
        samples_t_iv = []
        for s in range(num_samples):
            sample_norm = all_samples[t, s]
            sample_iv = preprocessor.inverse_transform(sample_norm[None, :, :])[0]
            samples_t_iv.append(sample_iv)
        samples_iv.append(np.array(samples_t_iv))
    
    samples_iv = np.array(samples_iv)  # [T, N_samples, 9, 9]
    targets_iv = preprocessor.inverse_transform(all_targets)  # [T, 9, 9]
    
    # Generate evaluation report
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)
    
    moneyness_labels = ['0.6', '0.7', '0.8', '0.9', '1.0', '1.1', '1.2', '1.3', '1.4']
    tenor_labels = ['1D', '1W', '2W', '1M', '2M', '3M', '6M', '9M', '1Y']
    
    report = generate_evaluation_report(
        samples=samples_iv,
        targets=targets_iv,
        moneyness_labels=moneyness_labels,
        tenor_labels=tenor_labels,
        moneyness_grid=data_dict['moneyness_grid'],
        tenor_grid=data_dict['tenor_grid']
    )
    
    return samples_iv, targets_iv, report


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================

def visualize_results(samples_iv, targets_iv, data_dict):
    """
    Create visualizations of results.
    """
    print("\n" + "=" * 60)
    print("STEP 5: VISUALIZATION")
    print("=" * 60)
    
    moneyness_grid = data_dict['moneyness_grid']
    tenor_grid = data_dict['tenor_grid']
    
    # 1. Surface comparison for a random day
    print("\n1. Plotting surface comparison...")
    random_idx = np.random.randint(0, len(targets_iv))
    plot_surface_comparison(
        real_surface=targets_iv[random_idx],
        pred_surface=samples_iv[random_idx].mean(axis=0),
        moneyness_grid=moneyness_grid,
        tenor_grid=tenor_grid,
        title=f"Surface Comparison (Test Day {random_idx})"
    )
    
    # 2. Time series for ATM 1-month
    print("\n2. Plotting time series for ATM 1-Month...")
    atm_idx = 4  # 1.0 moneyness
    tenor_idx = 3  # 1-month
    
    real_slice = targets_iv[:, atm_idx, tenor_idx]
    pred_mean = samples_iv[:, :, atm_idx, tenor_idx].mean(axis=1)
    pred_lower = np.percentile(samples_iv[:, :, atm_idx, tenor_idx], 5, axis=1)
    pred_upper = np.percentile(samples_iv[:, :, atm_idx, tenor_idx], 95, axis=1)
    
    plot_time_series_slice(
        real_data=real_slice,
        pred_mean=pred_mean,
        pred_lower=pred_lower,
        pred_upper=pred_upper,
        title="ATM 1-Month IV Forecast"
    )
    
    # 3. Distribution comparison
    print("\n3. Plotting distribution comparison...")
    plot_distribution_comparison(
        samples=samples_iv[:, :, atm_idx, tenor_idx].flatten(),
        targets=real_slice.flatten(),
        slice_name="ATM 1-Month"
    )
    
    print("\nVisualization complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Execute the complete pipeline.
    """
    print("\n" + "=" * 60)
    print("IV SURFACE DIFFUSION MODEL")
    print("Complete End-to-End Pipeline")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 50  # Increase for real training
    num_samples = 10
    
    # Step 1: Prepare data
    data_dict = prepare_synthetic_data(num_days=1500)
    
    # Step 2: Setup model
    model, schedule, ddpm = setup_model(device=device)
    
    # Step 3: Train (uncomment for actual training)
    train = input("\nTrain model? (y/n): ").lower() == 'y'
    if train:
        train_diffusion_model(
            model=model,
            ddpm=ddpm,
            data_dict=data_dict,
            device=device,
            num_epochs=num_epochs
        )
    else:
        print("\nSkipping training. Make sure 'best_model.pt' exists!")
    
    # Step 4: Evaluate
    eval_model = input("\nEvaluate model? (y/n): ").lower() == 'y'
    if eval_model:
        samples_iv, targets_iv, report = evaluate_model(
            model=model,
            ddpm=ddpm,
            data_dict=data_dict,
            device=device,
            num_samples=num_samples
        )
        
        # Step 5: Visualize
        visualize_results(samples_iv, targets_iv, data_dict)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Replace synthetic data with real OptionMetrics data")
    print("2. Increase training epochs (paper uses 2000)")
    print("3. Tune hyperparameters based on validation performance")
    print("4. Compare with VolGAN benchmark")


if __name__ == "__main__":
    main()
    # Note: Import statements at the top should be uncommented
    # when you have all modules in separate files
    
    print("="*60)
    print("COMPLETE EXAMPLE SCRIPT")
    print("="*60)
    print("\nThis script demonstrates the full pipeline:")
    print("1. Data preparation and preprocessing")
    print("2. Model setup (U-Net + DDPM)")
    print("3. Training with SNR-weighted arbitrage penalty")
    print("4. Sampling and evaluation")
    print("5. Visualization of results")
    print("\nTo run the complete pipeline, execute:")
    print("  python complete_example.py")
    print("\nMake sure all dependencies are installed:")
    print("  torch, numpy, scipy, pandas, matplotlib, tqdm")
