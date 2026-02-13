"""
Evaluation metrics for IV Surface Diffusion Model
Implements MAPE, CI breach rates, and arbitrage analysis from the paper
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# ============================================================================
# FORECASTING METRICS
# ============================================================================

def calculate_mape(
    predictions: np.ndarray,
    targets: np.ndarray,
    epsilon: float = 1e-8
) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Args:
        predictions: Predicted values [N]
        targets: True values [N]
        epsilon: Small constant to avoid division by zero
    
    Returns:
        MAPE as percentage
    """
    ape = np.abs((predictions - targets) / (targets + epsilon))
    mape = np.mean(ape) * 100
    return mape


def calculate_surface_mape(
    pred_surface: np.ndarray,
    true_surface: np.ndarray
) -> float:
    """
    Calculate MAPE for entire surface.
    
    Args:
        pred_surface: Predicted IV surface [H, W]
        true_surface: True IV surface [H, W]
    
    Returns:
        Surface MAPE as percentage
    """
    return calculate_mape(pred_surface.flatten(), true_surface.flatten())


def calculate_gridpoint_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    moneyness_labels: List[str],
    tenor_labels: List[str]
) -> pd.DataFrame:
    """
    Calculate MAPE and std(APE) for each grid point.
    
    Args:
        predictions: Predicted surfaces [T, H, W]
        targets: True surfaces [T, H, W]
        moneyness_labels: Labels for moneyness dimension
        tenor_labels: Labels for tenor dimension
    
    Returns:
        DataFrame with metrics per grid point
    """
    T, H, W = predictions.shape
    
    results = []
    for i, m_label in enumerate(moneyness_labels):
        for j, t_label in enumerate(tenor_labels):
            pred_slice = predictions[:, i, j]
            true_slice = targets[:, i, j]
            
            # Calculate APE for this slice
            ape = np.abs((pred_slice - true_slice) / (true_slice + 1e-8)) * 100
            
            results.append({
                'Moneyness': m_label,
                'Tenor': t_label,
                'MAPE (%)': np.mean(ape),
                'Std of APE (%)': np.std(ape)
            })
    
    return pd.DataFrame(results)


def calculate_confidence_interval_metrics(
    samples: np.ndarray,
    targets: np.ndarray,
    confidence_level: float = 0.90
) -> Dict[str, float]:
    """
    Calculate confidence interval metrics.
    
    Args:
        samples: Generated samples [T, N_samples, H, W]
        targets: True surfaces [T, H, W]
        confidence_level: Confidence level (e.g., 0.90 for 90% CI)
    
    Returns:
        Dictionary with CI metrics
    """
    T, N_samples, H, W = samples.shape
    
    # Calculate percentiles
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 - (1 - confidence_level) / 2) * 100
    
    lower_bound = np.percentile(samples, lower_percentile, axis=1)  # [T, H, W]
    upper_bound = np.percentile(samples, upper_percentile, axis=1)  # [T, H, W]
    
    # CI width
    ci_width = upper_bound - lower_bound
    mean_ci_width = np.mean(ci_width)
    std_ci_width = np.std(ci_width)
    
    # Breach rate (percentage of points outside CI)
    breaches = (targets < lower_bound) | (targets > upper_bound)
    breach_rate = np.mean(breaches) * 100
    
    return {
        'mean_ci_width': mean_ci_width,
        'std_ci_width': std_ci_width,
        'breach_rate_%': breach_rate,
        'expected_breach_%': (1 - confidence_level) * 100
    }


def calculate_gridpoint_ci_metrics(
    samples: np.ndarray,
    targets: np.ndarray,
    moneyness_labels: List[str],
    tenor_labels: List[str],
    confidence_level: float = 0.90
) -> pd.DataFrame:
    """
    Calculate CI metrics for each grid point.
    
    Args:
        samples: Generated samples [T, N_samples, H, W]
        targets: True surfaces [T, H, W]
        moneyness_labels: Labels for moneyness dimension
        tenor_labels: Labels for tenor dimension
        confidence_level: Confidence level
    
    Returns:
        DataFrame with CI metrics per grid point
    """
    T, N_samples, H, W = samples.shape
    
    lower_p = (1 - confidence_level) / 2 * 100
    upper_p = (1 - (1 - confidence_level) / 2) * 100
    
    results = []
    for i, m_label in enumerate(moneyness_labels):
        for j, t_label in enumerate(tenor_labels):
            sample_slice = samples[:, :, i, j]  # [T, N_samples]
            true_slice = targets[:, i, j]  # [T]
            
            # Calculate CI
            lower = np.percentile(sample_slice, lower_p, axis=1)
            upper = np.percentile(sample_slice, upper_p, axis=1)
            
            # Metrics
            ci_width = upper - lower
            breaches = (true_slice < lower) | (true_slice > upper)
            
            results.append({
                'Moneyness': m_label,
                'Tenor': t_label,
                'Mean CI Width': np.mean(ci_width),
                'Std CI Width': np.std(ci_width),
                'CI Breach %': np.mean(breaches) * 100
            })
    
    return pd.DataFrame(results)


# ============================================================================
# DISTRIBUTIONAL METRICS
# ============================================================================

def calculate_distribution_moments(
    samples: np.ndarray,
    targets: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Calculate first 4 statistical moments for samples vs targets.
    
    Args:
        samples: Generated samples [T, N_samples, H, W] or [T*N_samples, H, W]
        targets: True surfaces [T, H, W]
    
    Returns:
        Dictionary with moments for samples and targets
    """
    # Flatten all dimensions except the first (samples/time)
    if samples.ndim == 4:
        T, N, H, W = samples.shape
        samples_flat = samples.reshape(-1, H*W)
    else:
        samples_flat = samples.reshape(samples.shape[0], -1)
    
    targets_flat = targets.reshape(targets.shape[0], -1)
    
    # Pool all values
    samples_pooled = samples_flat.flatten()
    targets_pooled = targets_flat.flatten()
    
    moments = {
        'samples': {
            'mean': np.mean(samples_pooled),
            'std': np.std(samples_pooled),
            'skewness': stats.skew(samples_pooled),
            'kurtosis': stats.kurtosis(samples_pooled, fisher=True)
        },
        'targets': {
            'mean': np.mean(targets_pooled),
            'std': np.std(targets_pooled),
            'skewness': stats.skew(targets_pooled),
            'kurtosis': stats.kurtosis(targets_pooled, fisher=True)
        }
    }
    
    return moments


def calculate_slice_moments(
    samples: np.ndarray,
    targets: np.ndarray,
    slice_coords: Tuple[int, int],
    labels: Tuple[str, str]
) -> pd.DataFrame:
    """
    Calculate moments for a specific grid point slice.
    
    Args:
        samples: Generated samples [T, N_samples, H, W]
        targets: True surfaces [T, H, W]
        slice_coords: (i, j) coordinates of grid point
        labels: (moneyness_label, tenor_label)
    
    Returns:
        DataFrame comparing moments
    """
    i, j = slice_coords
    m_label, t_label = labels
    
    # Extract slice
    sample_slice = samples[:, :, i, j].flatten()
    target_slice = targets[:, i, j].flatten()
    
    moments_df = pd.DataFrame({
        'Metric': ['Mean', 'Std Dev', 'Skewness', 'Kurtosis (Fisher)'],
        'Real Data': [
            np.mean(target_slice),
            np.std(target_slice),
            stats.skew(target_slice),
            stats.kurtosis(target_slice, fisher=True)
        ],
        'Diffusion Model': [
            np.mean(sample_slice),
            np.std(sample_slice),
            stats.skew(sample_slice),
            stats.kurtosis(sample_slice, fisher=True)
        ]
    })
    
    moments_df['Location'] = f"{m_label} {t_label}"
    
    return moments_df


# ============================================================================
# ARBITRAGE ANALYSIS
# ============================================================================

def analyze_arbitrage_violations(
    surfaces: np.ndarray,
    moneyness_grid: np.ndarray,
    tenor_grid: np.ndarray,
    S: float = 100.0
) -> Dict[str, np.ndarray]:
    """
    Analyze arbitrage violations over time.
    
    Args:
        surfaces: IV surfaces [T, H, W]
        moneyness_grid: Moneyness values
        tenor_grid: Tenor values
        S: Spot price
    
    Returns:
        Dictionary with arbitrage penalties over time
    """
    T = surfaces.shape[0]
    
    # Import Black-Scholes function
    from iv_diffusion_main import black_scholes_call
    
    penalties = {
        'calendar': np.zeros(T),
        'call_spread': np.zeros(T),
        'butterfly': np.zeros(T),
        'total': np.zeros(T)
    }
    
    for t in range(T):
        surface = surfaces[t]  # [H, W]
        
        # Convert to torch for calculation
        surface_torch = torch.FloatTensor(surface).unsqueeze(0)  # [1, H, W]
        m_torch = torch.FloatTensor(moneyness_grid)
        tau_torch = torch.FloatTensor(tenor_grid)
        
        # Calculate penalties
        from iv_diffusion_main import calculate_arbitrage_penalty
        p1, p2, p3 = calculate_arbitrage_penalty(
            surface_torch, m_torch, tau_torch, S
        )
        
        penalties['calendar'][t] = p1.item()
        penalties['call_spread'][t] = p2.item()
        penalties['butterfly'][t] = p3.item()
        penalties['total'][t] = p1.item() + p2.item() + p3.item()
    
    return penalties


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_surface_comparison(
    real_surface: np.ndarray,
    pred_surface: np.ndarray,
    moneyness_grid: np.ndarray,
    tenor_grid: np.ndarray,
    title: str = "IV Surface Comparison",
    save_path: str = None
):
    """
    Plot 3D comparison of real vs predicted surface.
    
    Args:
        real_surface: Real IV surface [H, W]
        pred_surface: Predicted IV surface [H, W]
        moneyness_grid: Moneyness values
        tenor_grid: Tenor values
        title: Plot title
        save_path: Optional path to save figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    M, T = np.meshgrid(moneyness_grid, tenor_grid, indexing='ij')
    
    fig = plt.figure(figsize=(16, 6))
    
    # Real surface
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(M, T, real_surface, cmap='viridis', alpha=0.9)
    ax1.set_xlabel('Moneyness')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('Real Surface')
    
    # Predicted surface
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(M, T, pred_surface, cmap='viridis', alpha=0.9)
    ax2.set_xlabel('Moneyness')
    ax2.set_ylabel('Time to Maturity')
    ax2.set_zlabel('Implied Volatility')
    ax2.set_title('Predicted Surface (Mean)')
    
    # Difference
    ax3 = fig.add_subplot(133, projection='3d')
    diff = pred_surface - real_surface
    ax3.plot_surface(M, T, diff, cmap='RdBu', alpha=0.9)
    ax3.set_xlabel('Moneyness')
    ax3.set_ylabel('Time to Maturity')
    ax3.set_zlabel('Difference')
    ax3.set_title('Prediction Error')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_time_series_slice(
    real_data: np.ndarray,
    pred_mean: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
    dates: np.ndarray = None,
    title: str = "IV Forecast",
    save_path: str = None
):
    """
    Plot time series with confidence intervals.
    
    Args:
        real_data: Real IV values [T]
        pred_mean: Predicted mean [T]
        pred_lower: Lower CI bound [T]
        pred_upper: Upper CI bound [T]
        dates: Optional date array
        title: Plot title
        save_path: Optional save path
    """
    if dates is None:
        dates = np.arange(len(real_data))
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(dates, real_data, 'k-', label='Real', linewidth=1.5, alpha=0.8)
    plt.plot(dates, pred_mean, 'b-', label='Predicted Mean', linewidth=1.5)
    plt.fill_between(dates, pred_lower, pred_upper, 
                     color='blue', alpha=0.2, label='90% CI')
    
    plt.xlabel('Time')
    plt.ylabel('Implied Volatility')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_arbitrage_over_time(
    penalties: Dict[str, np.ndarray],
    dates: np.ndarray = None,
    title: str = "Arbitrage Penalties Over Time",
    save_path: str = None
):
    """
    Plot arbitrage penalties over time.
    
    Args:
        penalties: Dictionary with penalty arrays
        dates: Optional date array
        title: Plot title
        save_path: Optional save path
    """
    if dates is None:
        dates = np.arange(len(penalties['total']))
    
    plt.figure(figsize=(14, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(dates, penalties['total'], 'k-', linewidth=1.5, label='Total')
    plt.ylabel('Total Arbitrage Penalty')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(dates, penalties['calendar'], label='Calendar Spread', linewidth=1)
    plt.plot(dates, penalties['call_spread'], label='Call Spread', linewidth=1)
    plt.plot(dates, penalties['butterfly'], label='Butterfly', linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Penalty Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_distribution_comparison(
    samples: np.ndarray,
    targets: np.ndarray,
    slice_name: str = "Grid Point",
    save_path: str = None
):
    """
    Plot histogram comparison of distributions.
    
    Args:
        samples: Generated samples (flattened)
        targets: True values (flattened)
        slice_name: Name of the slice
        save_path: Optional save path
    """
    plt.figure(figsize=(10, 6))
    
    bins = 50
    plt.hist(targets, bins=bins, alpha=0.6, label='Real Data', 
             color='blue', density=True, edgecolor='black')
    plt.hist(samples, bins=bins, alpha=0.6, label='Diffusion Model', 
             color='orange', density=True, edgecolor='black')
    
    plt.xlabel('Implied Volatility')
    plt.ylabel('Density')
    plt.title(f'Distribution Comparison: {slice_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# COMPREHENSIVE EVALUATION REPORT
# ============================================================================

def generate_evaluation_report(
    samples: np.ndarray,
    targets: np.ndarray,
    moneyness_labels: List[str],
    tenor_labels: List[str],
    moneyness_grid: np.ndarray,
    tenor_grid: np.ndarray,
) -> Dict:
    """
    Generate comprehensive evaluation report.
    
    Args:
        samples: Generated samples [T, N_samples, H, W]
        targets: True surfaces [T, H, W]
        moneyness_labels: Labels for moneyness
        tenor_labels: Labels for tenor
        moneyness_grid: Moneyness values
        tenor_grid: Tenor values
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print("Generating Evaluation Report...")
    print("=" * 60)
    
    # Calculate mean predictions
    pred_mean = samples.mean(axis=1)  # [T, H, W]
    
    # 1. Overall MAPE
    overall_mape = calculate_surface_mape(pred_mean, targets)
    print(f"\n1. Overall MAPE: {overall_mape:.4f}%")
    
    # 2. Grid-point metrics
    print("\n2. Grid-point Metrics:")
    gridpoint_metrics = calculate_gridpoint_metrics(
        pred_mean, targets, moneyness_labels, tenor_labels
    )
    print(gridpoint_metrics.to_string())
    
    # 3. Confidence interval metrics
    print("\n3. Confidence Interval Metrics (90% CI):")
    ci_metrics = calculate_gridpoint_ci_metrics(
        samples, targets, moneyness_labels, tenor_labels
    )
    print(ci_metrics.to_string())
    
    # 4. Distributional metrics
    print("\n4. Distributional Metrics:")
    dist_moments = calculate_distribution_moments(samples, targets)
    print("\nReal Data:")
    for k, v in dist_moments['targets'].items():
        print(f"  {k}: {v:.4f}")
    print("\nGenerated Data:")
    for k, v in dist_moments['samples'].items():
        print(f"  {k}: {v:.4f}")
    
    # 5. Arbitrage analysis
    print("\n5. Arbitrage Analysis:")
    arb_pred = analyze_arbitrage_violations(pred_mean, moneyness_grid, tenor_grid)
    arb_real = analyze_arbitrage_violations(targets, moneyness_grid, tenor_grid)
    
    print(f"\nPredicted Surfaces:")
    print(f"  Mean total penalty: {np.mean(arb_pred['total']):.6f}")
    print(f"  Std total penalty: {np.std(arb_pred['total']):.6f}")
    
    print(f"\nReal Surfaces:")
    print(f"  Mean total penalty: {np.mean(arb_real['total']):.6f}")
    print(f"  Std total penalty: {np.std(arb_real['total']):.6f}")
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    
    return {
        'overall_mape': overall_mape,
        'gridpoint_metrics': gridpoint_metrics,
        'ci_metrics': ci_metrics,
        'distribution_moments': dist_moments,
        'arbitrage_pred': arb_pred,
        'arbitrage_real': arb_real
    }


if __name__ == "__main__":
    print("Evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("- calculate_mape: Point-wise MAPE calculation")
    print("- calculate_confidence_interval_metrics: CI breach rates")
    print("- calculate_distribution_moments: Statistical moments")
    print("- analyze_arbitrage_violations: Arbitrage penalty analysis")
    print("- generate_evaluation_report: Comprehensive evaluation")
    print("\nVisualization functions:")
    print("- plot_surface_comparison: 3D surface comparison")
    print("- plot_time_series_slice: Time series with CI")
    print("- plot_arbitrage_over_time: Arbitrage tracking")
    print("- plot_distribution_comparison: Distribution histograms")
