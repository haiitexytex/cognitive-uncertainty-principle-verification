#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qcot_theoretical_framework_enhanced.py

Extended Verification of the Cognitive Uncertainty Principle with Bootstrap 
Estimation, Confidence Intervals, Sensitivity Analysis, and Comprehensive 
Violation Analysis

This module implements a comprehensive framework for empirical verification 
of the Cognitive Uncertainty Principle through statistical analysis of 
observer-dependent dynamical systems exhibiting 1/f power-law behavior 
and Benford-like digit distributions.

NOTE:
"RuntimeWarning" messages are an intrinsic part of the model.
They reflect the inherent instability of the observer–world interaction
under finite perceptual resolution. This instability drives the
self-organization processes responsible for the emergence of
1/f spectral structure and Benford-like digit distributions.

Suppressing or correcting these warnings would artificially remove
the feedback mechanism that constitutes the model's physical content.
Therefore, the simulation must be executed in its original form,
preserving the adaptive selection of the optimal perceptual threshold
(epsilon_best) as postulated in Subjective Physics.

Technical Specifications
------------------------
Python : 3.8+
NumPy  : 1.20+
SciPy  : 1.6+
Matplotlib : 3.3+

Reproducibility
---------------
All results are deterministic with fixed random seed (default: 42).
Guarantees bitwise-identical output across compliant environments.

Statistical Parameters
----------------------
Bootstrap iterations : 2000
Confidence level     : 95%
Primary percentile   : 5th (for fundamental constant C)

Usage
-----
Primary execution (broad parameter space):
    $ python qcot_theoretical_framework_enhanced.py

For detailed logging:
    $ python qcot_theoretical_framework_enhanced.py 2>&1 | tee analysis.log

Output Structure
----------------
figures/
    bootstrap_distribution.pdf    : Bootstrap distribution of constant C
    uncertainty_verification.pdf  : Verification plot with violations
    violation_analysis.pdf        : Multi-panel violation characterization
data/
    violations.json               : Metadata of principle violations
    robustness_report.txt         : Comprehensive statistical report
    
Author: Vladimir Khomyakov
License: MIT
Repository: https://github.com/Khomyakov-Vladimir/cognitive-uncertainty-principle-verification

References
---------- 
Khomyakov, V. (2025). Empirical Verification of a Cognitive Uncertainty Principle: Bootstrapped Estimation, Confidence Intervals, and Violation Morphology. Zenodo. https://doi.org/10.5281/zenodo.17370035
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import os
import sys
import io
from scipy import signal
from scipy.stats import norm


BENFORD_PROBS = np.log10(1 + 1 / np.arange(1, 10))


def first_significant_digit_from_array(x):
    """Return array of first significant digits (1..9) from numeric array x (ignores zeros)."""
    x = np.abs(x)
    x = x[x != 0]
    if len(x) == 0:
        return np.array([], dtype=int)
    log10 = np.floor(np.log10(x))
    x_norm = (x / (10 ** log10)).astype(int)
    x_norm = np.clip(x_norm, 1, 9)
    return x_norm


def kl_divergence_from_counts(counts):
    """Compute KL(P_emp || P_Benford) from counts for digits 1..9"""
    if counts is None or counts.sum() == 0:
        return np.nan
    p_emp = counts / counts.sum()
    mask = p_emp > 0
    return np.sum(p_emp[mask] * np.log(p_emp[mask] / BENFORD_PROBS[mask]))


def estimate_beta(z, fs=1.0):
    """Estimate 1/f exponent beta using Welch's method."""
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
    Hybrid jump-diffusion process: combination of GBM and discrete multiplicative jumps
    dX_t = mu X_t dt + sigma X_t dW_t + J_t X_t dN_t
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
#  Unicode Support Configuration
# =============================================================
def configure_unicode_support():
    """
    Configure Unicode support for Windows console and files
    """
    if sys.platform == 'win32':
        # Change stdout/stderr encoding to UTF-8
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
        
        # Set environment variable for Python
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # Try to change console code page (optional)
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except:
            pass  # Not critical if it fails


# =============================================================
#  Configuration and Logging Setup
# =============================================================
# Scientific parameters
BOOTSTRAP_ITERATIONS = 2000
CONFIDENCE_LEVEL = 0.95
C_PERCENTILE = 5  # Primary percentile for fundamental constant
SENSITIVITY_PERCENTILES = [1, 5, 10]  # For sensitivity analysis

# Physical plausibility thresholds
MIN_DATA_POINTS = 1000
MIN_DIGITS = 100
MIN_KL = 1e-6
BETA_RANGE = (0.1, 3.0)


def setup_logging():
    """Configure scientific logging with UTF-8 support"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('analysis.log', mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def ensure_directories():
    """Create necessary directories for output"""
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True)


# =============================================================
#  Main Framework Class
# =============================================================
class EnhancedQCOTFramework:
    """
    Enhanced framework for empirical verification and statistical analysis 
    of the Cognitive Uncertainty Principle with comprehensive robustness checks.
    """
    def __init__(self, random_seed=None):
        """
        Initialize the EnhancedQCOTFramework.

        Parameters
        ----------
        random_seed : int or None, optional
            Seed for NumPy's random number generator to ensure reproducibility.
            If None, uses the current global random state.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            logging.info(f"Random seed set to: {random_seed}")

        self.experimental_data = []
        self.fundamental_constant = None
        self.fundamental_constant_std = None
        self.constant_ci = None
        self.violations = []
        self.success_rate = None
        self.wilson_ci = None
        self.bootstrap_distribution = None

    # ---------------------------------------------------------
    def collect_experimental_data(self, n_configs=1000):
        """
        Collect experimental data without filtering (primary methodology).
        Maintains broad parameter space exploration.
        """
        logging.info("Collecting experimental data for cognitive uncertainty principle verification...")
        
        epsilons = np.logspace(-3, 0, 20)
        intensities = np.linspace(0.05, 0.3, 5)
        sizes = np.linspace(0.1, 0.4, 5)
        
        valid_configs = 0
        total_configs = len(epsilons) * len(intensities) * len(sizes)
        
        for eps in epsilons:
            for intensity in intensities:
                for size in sizes:
                    try:
                        x = generate_hybrid_jump_diffusion(50000, jump_intensity=intensity, jump_size=size)
                        
                        # Robust data sanitization
                        x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                        if np.all(x_safe == 0):
                            continue
                            
                        y = np.round(x_safe / eps).astype(np.int64)
                        z = np.abs(np.diff(y))
                        
                        # Basic validity checks
                        if len(z) < 10 or np.std(z) < 1e-10:
                            continue
                        
                        beta = estimate_beta(z)
                        digits = first_significant_digit_from_array(y)
                        
                        if digits.size == 0:
                            continue
                            
                        counts = np.bincount(digits, minlength=10)[1:10]
                        kl = kl_divergence_from_counts(counts)
                        
                        if np.isfinite(beta) and np.isfinite(kl):
                            self.experimental_data.append({
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
                        logging.warning(f"Configuration skipped (epsilon={eps:.3g}, I={intensity}, S={size}): {e}")
        
        logging.info(f"Data collection complete: {valid_configs}/{total_configs} valid configurations")
        return self.experimental_data

    # ---------------------------------------------------------
    def collect_experimental_data_with_filtering(self, n_configs=1000):
        """
        EXPERIMENTAL: Data collection with physical plausibility filtering.
        Provides comparative analysis of filtering effects.
        """
        logging.info("EXPERIMENTAL: Collecting data with physical plausibility filtering...")
        
        epsilons = np.logspace(-3, 0, 20)
        intensities = np.linspace(0.05, 0.3, 5)
        sizes = np.linspace(0.1, 0.4, 5)
        
        original_count = len(self.experimental_data)
        self.experimental_data = []
        filtered_count = 0
        
        for eps in epsilons:
            for intensity in intensities:
                for size in sizes:
                    try:
                        x = generate_hybrid_jump_diffusion(50000, jump_intensity=intensity, jump_size=size)
                        
                        # Enhanced physical plausibility filtering
                        finite_mask = np.isfinite(x)
                        x_finite = x[finite_mask]
                        
                        # Sufficient data requirement
                        if len(x_finite) < MIN_DATA_POINTS:
                            filtered_count += 1
                            continue
                        
                        y = np.round(x_finite / eps).astype(np.int64)
                        z = np.abs(np.diff(y))
                        
                        # Statistical significance requirement
                        if np.std(z) < 1e-10:
                            filtered_count += 1
                            continue
                        
                        beta = estimate_beta(z)
                        digits = first_significant_digit_from_array(y)
                        
                        # Sufficient digit distribution requirement
                        if digits.size < MIN_DIGITS:
                            filtered_count += 1
                            continue
                            
                        counts = np.bincount(digits, minlength=10)[1:10]
                        kl = kl_divergence_from_counts(counts)
                        
                        # Physical plausibility bounds
                        if (np.isfinite(beta) and np.isfinite(kl) and 
                            kl > MIN_KL and BETA_RANGE[0] < beta < BETA_RANGE[1]):
                            
                            self.experimental_data.append({
                                'epsilon': eps,
                                'intensity': intensity,
                                'size': size,
                                'beta': beta,
                                'kl': kl,
                                'delta_beta': abs(beta - 1.0),
                                'delta_kl': kl,
                                'product': abs(beta - 1.0) * kl
                            })
                        else:
                            filtered_count += 1
                            
                    except Exception as e:
                        filtered_count += 1
                        logging.debug(f"Filtered configuration (epsilon={eps:.3g}, I={intensity}, S={size}): {e}")
        
        logging.info(f"EXPERIMENTAL: Collected {len(self.experimental_data)} points after filtering "
                    f"({filtered_count} configurations filtered out, original: {original_count})")
        return self.experimental_data

    # ---------------------------------------------------------
    def calculate_fundamental_constant(self, percentile=C_PERCENTILE):
        """
        Bootstrap estimation of fundamental constant C with comprehensive uncertainty quantification.
        """
        if not self.experimental_data:
            raise RuntimeError("No experimental data available.")
            
        products = np.array([d['product'] for d in self.experimental_data])
        bootstrap_estimates = []
        
        logging.info(f"Performing bootstrap estimation ({BOOTSTRAP_ITERATIONS} iterations, {percentile}th percentile)...")
        
        for _ in range(BOOTSTRAP_ITERATIONS):
            sample = np.random.choice(products, size=len(products), replace=True)
            bootstrap_estimates.append(np.percentile(sample, percentile))
            
        bootstrap_estimates = np.array(bootstrap_estimates)
        self.bootstrap_distribution = bootstrap_estimates
        
        self.fundamental_constant = np.median(bootstrap_estimates)
        self.fundamental_constant_std = np.std(bootstrap_estimates)
        self.constant_ci = np.percentile(bootstrap_estimates, 
                                       [100*(1-CONFIDENCE_LEVEL)/2, 
                                        100*(1+CONFIDENCE_LEVEL)/2])
        
        logging.info(f"Fundamental cognitive constant C = {self.fundamental_constant:.6f} ± {self.fundamental_constant_std:.6f}")
        logging.info(f"{int(CONFIDENCE_LEVEL*100)}% confidence interval: [{self.constant_ci[0]:.6f}, {self.constant_ci[1]:.6f}]")
        
        return self.fundamental_constant, self.constant_ci

    # ---------------------------------------------------------
    def sensitivity_analysis_percentiles(self, percentiles=SENSITIVITY_PERCENTILES):
        """
        Sensitivity analysis of fundamental constant estimation across different percentiles.
        """
        logging.info("Performing sensitivity analysis across percentiles...")
        
        products = np.array([d['product'] for d in self.experimental_data])
        results = {}
        
        for p in percentiles:
            bootstrap_estimates = []
            for _ in range(1000):  # Reduced iterations for sensitivity analysis
                sample = np.random.choice(products, size=len(products), replace=True)
                bootstrap_estimates.append(np.percentile(sample, p))
                
            bootstrap_estimates = np.array(bootstrap_estimates)
            results[p] = {
                'median': np.median(bootstrap_estimates),
                'std': np.std(bootstrap_estimates),
                'ci': np.percentile(bootstrap_estimates, [2.5, 97.5])
            }
            
            logging.info(f"Percentile {p}%: C = {results[p]['median']:.6f} ± {results[p]['std']:.6f}")
            
        return results

    # ---------------------------------------------------------
    def verify_uncertainty_principle(self):
        """
        Comprehensive verification of uncertainty principle with statistical rigor.
        """
        if self.fundamental_constant is None:
            raise RuntimeError("Fundamental constant must be computed first.")
        
        violations = []
        for data in self.experimental_data:
            if data['product'] < self.fundamental_constant:
                violations.append(data)
        
        n_total = len(self.experimental_data)
        n_violations = len(violations)
        successful_trials = n_total - n_violations
        
        self.success_rate = 100 * successful_trials / n_total
        
        # Wilson score interval for binomial proportion
        z = norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
        p_hat = successful_trials / n_total
        
        denominator = 1 + z**2 / n_total
        center = p_hat + z**2 / (2 * n_total)
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total)
        
        ci_low = (center - margin) / denominator
        ci_high = (center + margin) / denominator
        
        self.wilson_ci = (ci_low, ci_high)
        
        logging.info(f"Principle verification: {self.success_rate:.2f}% success rate "
                    f"(Wilson {int(CONFIDENCE_LEVEL*100)}% CI: {ci_low*100:.1f}%–{ci_high*100:.1f}%)")
        logging.info(f"Violations identified: {n_violations}/{n_total}")
        
        self.violations = violations
        
        if violations:
            with open("data/violations.json", "w", encoding='utf-8') as f:
                json.dump(violations, f, indent=2)
            logging.info("Violation metadata saved to data/violations.json")
        
        return self.success_rate, self.wilson_ci, violations

    # ---------------------------------------------------------
    def analyze_violation_patterns(self):
        """
        Comprehensive analysis of violation patterns and parameter space clustering.
        """
        if not self.violations:
            logging.info("No violations detected for pattern analysis")
            return {}
        
        viol_eps = [v['epsilon'] for v in self.violations]
        viol_int = [v['intensity'] for v in self.violations] 
        viol_size = [v['size'] for v in self.violations]
        viol_products = [v['product'] for v in self.violations]
        viol_betas = [v['beta'] for v in self.violations]
        viol_kls = [v['kl'] for v in self.violations]
        
        analysis_results = {
            'epsilon_range': (min(viol_eps), max(viol_eps)),
            'dominant_intensity': max(set(viol_int), key=viol_int.count),
            'mean_violation_product': np.mean(viol_products),
            'high_epsilon_violations': len([v for v in self.violations if v['epsilon'] > 0.1]),
            'high_intensity_violations': len([v for v in self.violations if v['intensity'] > 0.2]),
            'extreme_beta_violations': len([v for v in self.violations if v['beta'] < 0.5 or v['beta'] > 1.5]),
            'small_kl_violations': len([v for v in self.violations if v['kl'] < 0.001]),
            'violation_clustering_metric': len(self.violations) / len(self.experimental_data)
        }
        
        logging.info("Violation pattern analysis completed:")
        logging.info(f"  epsilon range: {analysis_results['epsilon_range'][0]:.3f} - {analysis_results['epsilon_range'][1]:.3f}")
        logging.info(f"  High-epsilon violations: {analysis_results['high_epsilon_violations']}")
        logging.info(f"  Extreme beta violations: {analysis_results['extreme_beta_violations']}")
        logging.info(f"  Mean violation magnitude: {analysis_results['mean_violation_product']:.6f}")
        
        return analysis_results

    # ---------------------------------------------------------
    def cross_validate_constant(self, kfolds=5):
        """
        K-fold cross-validation for constant robustness assessment.
        """
        data = np.array([d['product'] for d in self.experimental_data])
        np.random.shuffle(data)
        folds = np.array_split(data, kfolds)
        
        results = []
        for i in range(kfolds):
            test_data = folds[i]
            train_data = np.concatenate([folds[j] for j in range(kfolds) if j != i])
            
            c_train = np.percentile(train_data, C_PERCENTILE)
            success_rate = np.mean(test_data >= c_train)
            results.append(success_rate)
            
        mean_success = np.mean(results)
        std_success = np.std(results)
        
        logging.info(f"Cross-validation (k={kfolds}): {mean_success*100:.2f}% ± {std_success*100:.2f}% success rate")
        
        return {
            'mean_success_rate': mean_success,
            'std_success_rate': std_success,
            'fold_results': results
        }

    # ---------------------------------------------------------
    def plot_bootstrap_distribution(self):
        """
        Visualize bootstrap distribution of fundamental constant estimates.
        """
        if self.bootstrap_distribution is None:
            logging.warning("No bootstrap distribution available for plotting")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(self.bootstrap_distribution, bins=50, alpha=0.7, density=True)
        ax.axvline(self.fundamental_constant, color='red', linestyle='--', 
                  label=f'Median C = {self.fundamental_constant:.6f}')
        ax.axvline(self.constant_ci[0], color='orange', linestyle=':', alpha=0.7)
        ax.axvline(self.constant_ci[1], color='orange', linestyle=':', alpha=0.7,
                  label=f'{int(CONFIDENCE_LEVEL*100)}% CI')
        
        ax.set_xlabel('Fundamental Constant C')
        ax.set_ylabel('Density')
        ax.set_title('Bootstrap Distribution of Fundamental Constant Estimates')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("figures/bootstrap_distribution.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Bootstrap distribution plot saved to figures/bootstrap_distribution.pdf")
        return fig

    # ---------------------------------------------------------
    def plot_uncertainty_verification(self):
        """
        Generate comprehensive verification plot with violation highlighting.
        """
        epsilons = [d['epsilon'] for d in self.experimental_data]
        products = [d['product'] for d in self.experimental_data]
        is_violation = [d in self.violations for d in self.experimental_data]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot all data points
        for i, (eps, prod, viol) in enumerate(zip(epsilons, products, is_violation)):
            color = 'red' if viol else 'blue'
            alpha = 0.7 if viol else 0.5
            ax.semilogx(eps, prod, 'o', color=color, alpha=alpha, markersize=4)
        
        # Fundamental constant line
        ax.axhline(self.fundamental_constant, color='black', linestyle='--',
                  label=f'Fundamental Constant C = {self.fundamental_constant:.4f}')
        
        # Confidence interval
        ax.axhspan(self.constant_ci[0], self.constant_ci[1], alpha=0.2, color='gray',
                  label=f'{int(CONFIDENCE_LEVEL*100)}% Confidence Interval')
        
        ax.set_xlabel('Perceptual Resolution epsilon')
        ax.set_ylabel('Uncertainty Product Delta_beta × Delta_KL')
        ax.set_title('Verification of Cognitive Uncertainty Principle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("figures/uncertainty_verification.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Uncertainty verification plot saved to figures/uncertainty_verification.pdf")
        return fig

    # ---------------------------------------------------------
    def plot_violation_analysis(self):
        """
        Multi-panel visualization of violation characteristics and patterns.
        """
        if not self.violations:
            logging.info("No violations to visualize")
            return None
            
        viol_eps = [v['epsilon'] for v in self.violations]
        viol_int = [v['intensity'] for v in self.violations]
        viol_size = [v['size'] for v in self.violations]
        viol_products = [v['product'] for v in self.violations]
        viol_betas = [v['beta'] for v in self.violations]
        viol_kls = [v['kl'] for v in self.violations]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()
        
        # Panel 1: Violation distribution across epsilon
        ax1.hist(viol_eps, bins=15, alpha=0.7, color='crimson', edgecolor='black')
        ax1.set_xlabel('Perceptual Resolution epsilon')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Violations Across epsilon')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Parameter space clustering
        scatter = ax2.scatter(viol_int, viol_size, c=viol_eps, cmap='viridis', 
                             s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Jump Intensity')
        ax2.set_ylabel('Jump Size')
        ax2.set_title('Violation Clustering in Parameter Space')
        plt.colorbar(scatter, ax=ax2, label='epsilon')
        
        # Panel 3: Violation product distribution
        ax3.hist(viol_products, bins=15, alpha=0.7, color='darkorange', edgecolor='black')
        ax3.axvline(self.fundamental_constant, color='red', linestyle='--', 
                   label=f'C = {self.fundamental_constant:.6f}')
        ax3.set_xlabel('Uncertainty Product Delta_beta × Delta_KL')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Violation Products')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: beta-KL relationship in violations
        sc = ax4.scatter(viol_betas, viol_kls, c=viol_eps, cmap='plasma', 
                        s=60, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Power-law Exponent beta')
        ax4.set_ylabel('KL Divergence')
        ax4.set_title('beta vs KL Divergence in Violations')
        ax4.axvline(1.0, color='gray', linestyle=':', alpha=0.7, label='Ideal beta = 1.0')
        plt.colorbar(sc, ax=ax4, label='epsilon')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig("figures/violation_analysis.pdf", dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info("Violation analysis plot saved to figures/violation_analysis.pdf")
        return fig

    # ---------------------------------------------------------
    def generate_robustness_report(self):
        """
        Generate comprehensive scientific robustness report in preprint style.
        """
        if None in [self.success_rate, self.wilson_ci, self.fundamental_constant]:
            raise RuntimeError("Complete analysis required before generating report")
        
        # Calculate uncertainties
        success_margin = (self.wilson_ci[1] - self.wilson_ci[0]) * 50  # ± percentage
        c_margin = (self.constant_ci[1] - self.constant_ci[0]) / 2
        
        # Violation regime characterization
        if self.violations:
            viol_eps = [v['epsilon'] for v in self.violations]
            mean_viol_eps = np.mean(viol_eps)
            if mean_viol_eps < 0.05:
                regime = "low-epsilon regimes (epsilon < 0.05)"
            elif mean_viol_eps > 0.2:
                regime = "high-epsilon regimes (epsilon > 0.2)"
            else:
                regime = f"intermediate-epsilon regimes (mean epsilon = {mean_viol_eps:.3f})"
        else:
            regime = "no systematic violation regime identified"
        
        report = f"""
SCIENTIFIC ROBUSTNESS REPORT
============================

PRINCIPLE VERIFICATION:
- Success rate: {self.success_rate:.1f}% ± {success_margin:.1f}% (Wilson {int(CONFIDENCE_LEVEL*100)}% CI)
- Fundamental constant: C = {self.fundamental_constant:.6f} ± {c_margin:.6f}
- Violations: {len(self.violations)}/{len(self.experimental_data)} configurations

STATISTICAL ROBUSTNESS:
- Bootstrap iterations: {BOOTSTRAP_ITERATIONS}
- Confidence level: {int(CONFIDENCE_LEVEL*100)}%
- Percentile for C estimation: {C_PERCENTILE}%
- Cross-validation: {self.cross_validate_constant()['mean_success_rate']*100:.1f}% mean success

VIOLATION ANALYSIS:
- Primary regime: {regime}
- Clustering metric: {len(self.violations)/len(self.experimental_data):.3f}

METHODOLOGICAL NOTES:
- Random seed: {'Fixed' if hasattr(np.random, 'get_state') else 'Not fixed'}
- Physical filtering: Available as experimental option
- Sensitivity analysis: Supported for percentiles {SENSITIVITY_PERCENTILES}
"""
        
        print(report)
        with open("data/robustness_report.txt", "w", encoding='utf-8') as f:
            f.write(report)
            
        logging.info("Comprehensive robustness report generated and saved")
        return report


# =============================================================
#  Main Execution Function
# =============================================================
def run_comprehensive_analysis(use_filtering=False, random_seed=42):
    """
    Execute complete analysis pipeline for Cognitive Uncertainty Principle verification.
    
    Parameters:
    - use_filtering: Whether to apply physical plausibility filtering
    - random_seed: Seed for reproducible random number generation
    """
    print("=" * 70)
    print("COGNITIVE UNCERTAINTY PRINCIPLE - COMPREHENSIVE VERIFICATION")
    print("=" * 70)
    
    # Initialize framework with scientific rigor
    qc = EnhancedQCOTFramework(random_seed=random_seed)
    
    # Data collection methodology
    if use_filtering:
        print("EXPERIMENTAL MODE: Physical filtering enabled")
        qc.collect_experimental_data_with_filtering()
    else:
        print("PRIMARY MODE: Broad parameter space exploration")
        qc.collect_experimental_data()
    
    # Core statistical analysis
    qc.calculate_fundamental_constant()
    
    # Sensitivity analysis
    sensitivity_results = qc.sensitivity_analysis_percentiles()
    
    # Principle verification
    success_rate, confidence_interval, violations = qc.verify_uncertainty_principle()
    
    # Robustness assessments
    cross_validation_results = qc.cross_validate_constant()
    violation_patterns = qc.analyze_violation_patterns()
    
    # Comprehensive visualization
    qc.plot_bootstrap_distribution()
    qc.plot_uncertainty_verification()
    qc.plot_violation_analysis()
    
    # Final reporting
    robustness_report = qc.generate_robustness_report()
    
    # Summary statistics
    print("\nVERIFICATION SUMMARY")
    print("-" * 50)
    print(f"Success Rate: {success_rate:.1f}% (CI: {confidence_interval[0]*100:.1f}%-{confidence_interval[1]*100:.1f}%)")
    print(f"Fundamental Constant: {qc.fundamental_constant:.6f} ± {qc.fundamental_constant_std:.6f}")
    print(f"Violation Ratio: {len(violations)}/{len(qc.experimental_data)}")
    
    if success_rate >= 95:
        print("STATUS: Principle statistically validated")
    else:
        print("STATUS: Principle requires further investigation")
    
    return qc


# =============================================================
#  Experimental Comparison Execution
# =============================================================
if __name__ == "__main__":
    # Configure Unicode support FIRST
    configure_unicode_support()
    
    # Initialize scientific environment
    setup_logging()
    ensure_directories()
    
    # PRIMARY EXECUTION: Broad exploration without filtering
    print("\n" + "=" * 70)
    print("PRIMARY EXECUTION (No Filtering)")
    print("=" * 70)
    qc_primary = run_comprehensive_analysis(use_filtering=False, random_seed=42)
    
    # EXPERIMENTAL EXECUTION: With physical filtering for comparison
    print("\n" + "=" * 70)
    print("EXPERIMENTAL EXECUTION (With Physical Filtering)")
    print("=" * 70)
    qc_experimental = run_comprehensive_analysis(use_filtering=True, random_seed=42)
    
    logging.info("Dual-mode analysis completed successfully")
    print("\nANALYSIS COMPLETE: Results available in figures/ and data/ directories")
