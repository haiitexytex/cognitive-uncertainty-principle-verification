#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kl_tradeoff_analysis.py

Perceptual Uncertainty Analysis via Epsilon-Aggregation

This module implements a complementary analysis of the Cognitive Uncertainty 
Principle by examining the perceptual resolution trade-off. While the canonical 
formulation establishes a fundamental bound on the product Δβ × KL(ε), this 
analysis investigates the dual relationship Δε × KL(ε) through Fisher 
information-based uncertainty quantification.

The methodology employs robust median aggregation across parameter configurations 
to extract the intrinsic KL divergence dependence on perceptual resolution ε, 
followed by spline-based differentiation in log-log space to compute Fisher 
information and corresponding uncertainty bounds.

Theoretical Framework
---------------------
Canonical uncertainty relation:
    Δβ × KL(ε) ≥ C_β

Perceptual uncertainty relation (investigated here):
    Δε × KL(ε) ≥ C_ε

where:
    - Δβ: spectral uncertainty (deviation from ideal 1/f scaling)
    - Δε: perceptual resolution uncertainty (Fisher information-based)
    - KL(ε): Kullback-Leibler divergence from Benford's law
    - C_β, C_ε: fundamental constants of cognitive observation

Technical Specifications
------------------------
Python : 3.8+
NumPy  : 1.20+
SciPy  : 1.6+
Matplotlib : 3.3+

Statistical Methods
-------------------
- Robust median aggregation for KL(ε) estimation
- Univariate spline smoothing in log-log space
- Fisher information-based uncertainty quantification
- Percentile-based constant estimation (5th percentile)

Output Structure
----------------
data/
    kl_tradeoff_extended.npz : Numerical results and uncertainty estimates
figures/
    canonical_relation.pdf : Canonical uncertainty relation figure
    perceptual_relation.pdf : Perceptual uncertainty relation figure

Usage
-----
Primary execution:
    $ python kl_tradeoff_analysis.py

For integration with main framework:
    >>> from kl_tradeoff_analysis import analyze_perceptual_tradeoff
    >>> results = analyze_perceptual_tradeoff(experimental_data)

Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/cognitive-uncertainty-principle-verification

References
----------
Khomyakov, V. (2025). Cognitive Uncertainty Principle: Dual Empirical–Theoretical Verification via Jump–Diffusion Dynamics. Zenodo. https://doi.org/10.5281/zenodo.17370034
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import sys
from scipy import signal
from scipy.interpolate import UnivariateSpline


# =============================================================
#  Universal Path Configuration
# =============================================================
def configure_output_paths():
    """
    Configure output paths for figures and data to be saved in root repository directories.
    Works from both root and scripts/ subdirectory.
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if we're in scripts/ subdirectory
    if os.path.basename(script_dir) == 'scripts':
        # Move up to root directory
        root_dir = os.path.dirname(script_dir)
    else:
        # We're already in root directory
        root_dir = script_dir
    
    # Create figures and data directories in root if they don't exist
    figures_dir = os.path.join(root_dir, 'figures')
    data_dir = os.path.join(root_dir, 'data')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    return figures_dir, data_dir

# Configure paths before any other imports
FIGURES_DIR, DATA_DIR = configure_output_paths()


# =============================================================
#  Core Computational Functions
# =============================================================

BENFORD_PROBS = np.log10(1 + 1 / np.arange(1, 10))


def first_significant_digit_from_array(x):
    """
    Extract first significant digits from numeric array.
    
    Parameters
    ----------
    x : ndarray
        Input numeric array
        
    Returns
    -------
    ndarray
        Array of first significant digits (1-9), empty if no valid values
    """
    x = np.abs(x)
    x = x[x != 0]
    if len(x) == 0:
        return np.array([], dtype=int)
    log10 = np.floor(np.log10(x))
    x_norm = (x / (10 ** log10)).astype(int)
    x_norm = np.clip(x_norm, 1, 9)
    return x_norm


def kl_divergence_from_counts(counts):
    """
    Compute KL divergence from Benford's law.
    
    Parameters
    ----------
    counts : ndarray
        Frequency counts for digits 1-9
        
    Returns
    -------
    float
        KL(P_empirical || P_Benford)
    """
    if counts is None or counts.sum() == 0:
        return np.nan
    p_emp = counts / counts.sum()
    mask = p_emp > 0
    return np.sum(p_emp[mask] * np.log(p_emp[mask] / BENFORD_PROBS[mask]))


def estimate_beta(z, fs=1.0):
    """
    Estimate 1/f power-law exponent using Welch's method.
    
    Parameters
    ----------
    z : ndarray
        Time series data
    fs : float, optional
        Sampling frequency (default: 1.0)
        
    Returns
    -------
    float
        Estimated spectral exponent beta
    """
    nperseg = 2**12
    if len(z) < nperseg:
        return np.nan
    f, Pxx = signal.welch(
        z, fs=fs, window='hann', nperseg=nperseg,
        noverlap=nperseg // 2, scaling='density'
    )
    mask = (f >= 1e-3) & (f <= 1e-1) & (Pxx > 0)
    if np.sum(mask) < 10:
        return np.nan
    log_f = np.log10(f[mask])
    log_Pxx = np.log10(Pxx[mask])
    slope, _ = np.polyfit(log_f, log_Pxx, 1)
    return -slope


def generate_hybrid_jump_diffusion(T, mu=0.0, sigma=0.01, jump_intensity=0.05, jump_size=0.2):
    """
    Generate hybrid jump-diffusion process.
    
    Implements stochastic differential equation:
        dX_t = mu X_t dt + sigma X_t dW_t + J_t X_t dN_t
    
    Parameters
    ----------
    T : int
        Number of time steps
    mu : float, optional
        Drift coefficient (default: 0.0)
    sigma : float, optional
        Diffusion coefficient (default: 0.01)
    jump_intensity : float, optional
        Poisson jump rate (default: 0.05)
    jump_size : float, optional
        Log-normal jump volatility (default: 0.2)
        
    Returns
    -------
    ndarray
        Simulated trajectory
    """
    x = np.ones(T)
    dt = 1.0
    
    for t in range(1, T):
        dW = np.random.normal(0, np.sqrt(dt))
        x_gbm = x[t-1] * (1 + mu * dt + sigma * dW)
        
        if np.random.random() < jump_intensity:
            jump_multiplier = np.random.lognormal(mean=0, sigma=jump_size)
            x[t] = x_gbm * jump_multiplier
        else:
            x[t] = x_gbm
            
    return x


# =============================================================
#  Data Collection Framework
# =============================================================

def collect_experimental_data(n_configs=1000, random_seed=42):
    """
    Collect experimental data for uncertainty analysis.
    
    Parameters
    ----------
    n_configs : int, optional
        Target number of configurations (actual may vary)
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    list of dict
        Experimental measurements with keys: epsilon, intensity, size, 
        beta, kl, delta_beta, delta_kl, product
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    logging.info("Collecting experimental data...")
    
    epsilons = np.logspace(-3, 0, 20)
    intensities = np.linspace(0.05, 0.3, 5)
    sizes = np.linspace(0.1, 0.4, 5)
    
    experimental_data = []
    valid_configs = 0
    
    for eps in epsilons:
        for intensity in intensities:
            for size in sizes:
                try:
                    x = generate_hybrid_jump_diffusion(
                        50000, jump_intensity=intensity, jump_size=size
                    )
                    
                    x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                    if np.all(x_safe == 0):
                        continue
                        
                    y = np.round(x_safe / eps).astype(np.int64)
                    z = np.abs(np.diff(y))
                    
                    if len(z) < 10 or np.std(z) < 1e-10:
                        continue
                    
                    beta = estimate_beta(z)
                    digits = first_significant_digit_from_array(y)
                    
                    if digits.size == 0:
                        continue
                        
                    counts = np.bincount(digits, minlength=10)[1:10]
                    kl = kl_divergence_from_counts(counts)
                    
                    if np.isfinite(beta) and np.isfinite(kl):
                        experimental_data.append({
                            'epsilon': eps,
                            'intensity': intensity,
                            'size': size,
                            'beta': beta,
                            'kl': kl,
                            'delta_beta': abs(beta - 1.0),
                            'delta_kl': kl,
                            'product': abs(beta - 1.0) * kl
                        })
                        valid_configs += 1
                        
                except Exception as e:
                    logging.debug(f"Configuration skipped: {e}")
    
    logging.info(f"Collected {valid_configs} valid configurations")
    return experimental_data


def calculate_fundamental_constant(experimental_data, percentile=5, 
                                   bootstrap_iterations=2000):
    """
    Bootstrap estimation of fundamental constant.
    
    Parameters
    ----------
    experimental_data : list of dict
        Experimental measurements
    percentile : float, optional
        Percentile for constant estimation (default: 5)
    bootstrap_iterations : int, optional
        Number of bootstrap samples (default: 2000)
        
    Returns
    -------
    tuple
        (fundamental_constant, confidence_interval)
    """
    products = np.array([d['product'] for d in experimental_data])
    bootstrap_estimates = []
    
    logging.info(f"Performing bootstrap estimation ({bootstrap_iterations} iterations)...")
    
    for _ in range(bootstrap_iterations):
        sample = np.random.choice(products, size=len(products), replace=True)
        bootstrap_estimates.append(np.percentile(sample, percentile))
        
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    constant = np.median(bootstrap_estimates)
    ci = np.percentile(bootstrap_estimates, [2.5, 97.5])
    
    logging.info(f"Fundamental constant C_beta = {constant:.6f} (95% CI: [{ci[0]:.6f}, {ci[1]:.6f}])")
    
    return constant, ci


# =============================================================
#  Perceptual Trade-off Analysis
# =============================================================

def analyze_perceptual_tradeoff(experimental_data, fundamental_constant):
    """
    Analyze perceptual uncertainty trade-off via epsilon-aggregation.
    
    This function implements the core analytical methodology:
    1. Robust median aggregation of KL(epsilon) across parameter configurations
    2. Spline-based differentiation in log-log space
    3. Fisher information computation
    4. Perceptual uncertainty bounds
    
    Parameters
    ----------
    experimental_data : list of dict
        Experimental measurements from data collection
    fundamental_constant : float
        Canonical fundamental constant C_beta for comparison
        
    Returns
    -------
    dict
        Analysis results containing:
        - epsilon: sorted epsilon values
        - kl_spline: smoothed KL divergence
        - delta_eps: perceptual uncertainty bounds
        - product_eps_kl: uncertainty products
        - C_eps: perceptual fundamental constant
        - C_beta: canonical fundamental constant (for reference)
    """
    logging.info("Analyzing perceptual trade-off...")
    
    # Extract raw data
    epsilons_raw = np.array([d['epsilon'] for d in experimental_data])
    kl_raw = np.array([d['kl'] for d in experimental_data])
    
    # Robust median aggregation
    unique_eps = np.unique(epsilons_raw)
    kl_agg = []
    
    for eps in unique_eps:
        mask = np.isclose(epsilons_raw, eps, rtol=1e-5)
        kl_vals = kl_raw[mask]
        kl_agg.append(np.nanmedian(kl_vals))
    
    kl_agg = np.array(kl_agg)
    
    # Remove invalid values
    valid = np.isfinite(kl_agg)
    eps_clean = unique_eps[valid]
    kl_clean = kl_agg[valid]
    
    if len(eps_clean) < 5:
        raise RuntimeError("Insufficient valid epsilon points after aggregation")
    
    # Sort for monotonicity
    sort_idx = np.argsort(eps_clean)
    eps_sorted = eps_clean[sort_idx]
    kl_sorted = kl_clean[sort_idx]
    
    # Spline smoothing in log-log space
    log_eps = np.log10(eps_sorted)
    log_kl = np.log10(np.clip(kl_sorted, 1e-12, None))
    
    spline = UnivariateSpline(log_eps, log_kl, s=0.5 * len(log_eps))
    dlogkl_dlogeps = spline.derivative()(log_eps)
    
    kl_spline = 10 ** spline(log_eps)
    eps_spline = 10 ** log_eps
    
    # Transform derivative to linear space
    dkl_deps = dlogkl_dlogeps * (kl_spline / eps_spline)
    
    # Fisher information and uncertainty bounds
    fisher = dkl_deps ** 2
    delta_eps = np.where(fisher > 0, 1.0 / np.sqrt(fisher), np.nan)
    
    # Perceptual uncertainty product
    product_eps_kl = delta_eps * kl_spline
    
    # Estimate perceptual constant
    valid_prod = np.isfinite(product_eps_kl) & (product_eps_kl > 0)
    C_eps = np.nan
    
    if np.any(valid_prod):
        C_eps = np.percentile(product_eps_kl[valid_prod], 5)
    
    logging.info(f"Canonical C_beta = {fundamental_constant:.6f}")
    logging.info(f"Perceptual C_epsilon = {C_eps:.6f}")
    
    return {
        'epsilon': eps_sorted,
        'kl_spline': kl_spline,
        'delta_eps': delta_eps,
        'product_eps_kl': product_eps_kl,
        'C_eps': C_eps,
        'C_beta': fundamental_constant
    }


# =============================================================
#  Visualization
# =============================================================

def plot_canonical_relation(experimental_data, C_beta):
    """
    Generate standalone visualization of canonical uncertainty relation.
    
    Creates publication-ready figure showing the empirical verification of
    the canonical uncertainty principle Δβ × KL(ε) ≥ C_β across the full
    parameter space.
    
    Parameters
    ----------
    experimental_data : list of dict
        Raw experimental measurements
    C_beta : float
        Canonical fundamental constant
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure object
    """
    logging.info("Generating canonical relation figure...")
    
    # Extract data
    epsilons_raw = np.array([d['epsilon'] for d in experimental_data])
    beta_raw = np.array([d['beta'] for d in experimental_data])
    kl_raw = np.array([d['kl'] for d in experimental_data])
    product_beta_kl = np.abs(beta_raw - 1.0) * kl_raw
    
    # Create figure with publication dimensions
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot measurements
    ax.scatter(epsilons_raw, product_beta_kl, c='steelblue', s=12, alpha=0.5, 
               label='Experimental measurements', edgecolors='none')
    
    # Plot fundamental constant
    ax.axhline(C_beta, color='crimson', linestyle='--', linewidth=2.5,
               label=f"$C_\\beta = {C_beta:.4f}$")
    
    # Styling
    ax.set_xscale('log')
    ax.set_xlabel(r'Perceptual Resolution $\epsilon$', fontsize=14)
    ax.set_ylabel(r'$\Delta\beta \cdot \mathrm{KL}(\epsilon)$', fontsize=14)
    ax.set_title('Canonical Cognitive Uncertainty Relation', fontsize=15, pad=15)
    ax.legend(frameon=True, fancybox=True, fontsize=12, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    
    # Save figure using universal path
    output_path = os.path.join(FIGURES_DIR, 'canonical_relation.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Canonical relation figure saved to {output_path}")
    
    return fig


def plot_perceptual_relation(analysis_results):
    """
    Generate standalone visualization of perceptual uncertainty relation.
    
    Creates publication-ready figure showing the perceptual uncertainty
    principle Δε × KL(ε) ≥ C_ε derived from Fisher information analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Results from perceptual trade-off analysis containing:
        - epsilon: epsilon values
        - product_eps_kl: uncertainty products
        - C_eps: perceptual fundamental constant
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure object
    """
    logging.info("Generating perceptual relation figure...")
    
    # Create figure with publication dimensions
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if np.isfinite(analysis_results['C_eps']):
        # Plot perceptual uncertainty product
        ax.plot(analysis_results['epsilon'], analysis_results['product_eps_kl'], 
                'o-', color='darkorange', markersize=8, linewidth=2.5, 
                markeredgecolor='darkred', markeredgewidth=1.0,
                label='Perceptual uncertainty product')
        
        # Plot fundamental constant
        ax.axhline(analysis_results['C_eps'], color='crimson', linestyle='--', 
                   linewidth=2.5,
                   label=f"$C_\\epsilon = {analysis_results['C_eps']:.4f}$")
        
        # Styling
        ax.set_xscale('log')
        ax.set_xlabel(r'Perceptual Resolution $\epsilon$', fontsize=14)
        ax.set_ylabel(r'$\Delta\epsilon \cdot \mathrm{KL}(\epsilon)$', fontsize=14)
        ax.set_title('Perceptual Cognitive Uncertainty Relation', fontsize=15, pad=15)
        ax.legend(frameon=True, fancybox=True, fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=12)
        
    else:
        # Handle undefined constant case
        ax.text(0.5, 0.5, 'Perceptual constant undefined\n(insufficient Fisher information)', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(r'Perceptual Resolution $\epsilon$', fontsize=14)
        ax.set_ylabel(r'$\Delta\epsilon \cdot \mathrm{KL}(\epsilon)$', fontsize=14)
        ax.set_title('Perceptual Cognitive Uncertainty Relation', fontsize=15, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save figure using universal path
    output_path = os.path.join(FIGURES_DIR, 'perceptual_relation.pdf')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    logging.info(f"Perceptual relation figure saved to {output_path}")
    
    return fig


# =============================================================
#  Main Execution
# =============================================================

def main():
    """
    Execute complete perceptual trade-off analysis pipeline.
    
    Pipeline stages:
    1. Data collection via hybrid jump-diffusion simulations
    2. Canonical fundamental constant estimation
    3. Perceptual trade-off analysis via epsilon-aggregation
    4. Publication-ready visualization (two separate figures)
    5. Result persistence
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("PERCEPTUAL UNCERTAINTY ANALYSIS")
    print("=" * 70)
    
    # Stage 1: Data collection
    print("\nStage 1: Collecting experimental data...")
    experimental_data = collect_experimental_data(random_seed=42)
    
    if len(experimental_data) < 100:
        raise RuntimeError(f"Insufficient data: only {len(experimental_data)} configurations")
    
    # Stage 2: Canonical constant estimation
    print("\nStage 2: Estimating canonical fundamental constant...")
    fundamental_constant, ci = calculate_fundamental_constant(experimental_data)
    
    # Stage 3: Perceptual trade-off analysis
    print("\nStage 3: Analyzing perceptual trade-off...")
    analysis_results = analyze_perceptual_tradeoff(experimental_data, fundamental_constant)
    
    # Stage 4: Visualization (two separate publication-ready figures)
    print("\nStage 4: Generating publication-ready figures...")
    fig1 = plot_canonical_relation(experimental_data, fundamental_constant)
    fig2 = plot_perceptual_relation(analysis_results)
    
    # Stage 5: Save results using universal paths
    print("\nStage 5: Saving results...")
    
    data_output_path = os.path.join(DATA_DIR, 'kl_tradeoff_extended.npz')
    np.savez(data_output_path,
             epsilon=analysis_results['epsilon'],
             kl=analysis_results['kl_spline'],
             delta_eps=analysis_results['delta_eps'],
             product_eps_kl=analysis_results['product_eps_kl'],
             C_eps=analysis_results['C_eps'],
             C_beta=analysis_results['C_beta'])
    
    # Close figures
    plt.close(fig1)
    plt.close(fig2)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Configurations analyzed: {len(experimental_data)}")
    print(f"Canonical constant:      C_beta = {fundamental_constant:.6f}")
    print(f"Perceptual constant:     C_epsilon = {analysis_results['C_eps']:.6f}")
    print("\nResults saved to:")
    print(f"  - {os.path.join(DATA_DIR, 'kl_tradeoff_extended.npz')}")
    print(f"  - {os.path.join(FIGURES_DIR, 'canonical_relation.pdf')}    (Figure 1)")
    print(f"  - {os.path.join(FIGURES_DIR, 'perceptual_relation.pdf')}   (Figure 2)")
    print("=" * 70)


if __name__ == "__main__":
    main()
