# Reproducibility Package
**Empirical Verification of a Cognitive Uncertainty Principle: Bootstrapped Estimation, Confidence Intervals, and Violation Morphology**

---

## Overview

> This reproducibility package corresponds exactly to the archived Zenodo version and ensures full computational transparency. All results can be regenerated directly from the provided scripts and data files.
This package provides the Python scripts used in the publication:

*Vladimir Khomyakov (2025). Empirical Verification of a Cognitive Uncertainty Principle: Bootstrapped Estimation, Confidence Intervals, and Violation Morphology.*

- **Version-specific DOI:** [10.5281/zenodo.17370035](https://doi.org/10.5281/zenodo.17370035)  
- **Concept DOI (latest version):** [10.5281/zenodo.17370034](https://doi.org/10.5281/zenodo.17370034) 

---

## Description (for Zenodo)

This work presents the first empirical verification of a Cognitive Uncertainty Principle (CUP) within the framework of Subjective Physics. The principle posits a fundamental lower bound on the product of two observer-centric uncertainty measures: (i) deviation of the spectral exponent (Δβ) from the ideal 1/f noise regime, and (ii) divergence (ΔKL) of leading-digit statistics from Benford's law. Using a large-scale simulation of synthetic observers based on a hybrid jump-diffusion model, we estimate a robust empirical constant C≈3.94×10⁻⁴ such that Δβ×ΔKL≳C holds in 95% of configurations. Violations cluster in high-perceptual-resolution regimes, suggesting a boundary of cognitive stability. The study introduces a reproducible pipeline for bootstrap-based confidence estimation and provides open data and code for full verification. This result supports the view that observer entropy and perceptual resolution are fundamentally coupled, offering a quantitative bridge between information-theoretic cognition and statistical physics.

---

## Repository
- **Source repository:** [https://github.com/Khomyakov-Vladimir/cognitive-uncertainty-principle-verification](https://github.com/Khomyakov-Vladimir/cognitive-uncertainty-principle-verification)

---

## Package Structure

```
cognitive-uncertainty-principle-verification/
│
├── README.md
│
├── scripts/
│   │ 
│   └── qcot_theoretical_framework_enhanced.py     # Extended Verification of the Cognitive Uncertainty Principle with Bootstrap
│                                                  # Estimation, Confidence Intervals, Sensitivity Analysis, and Comprehensive 
│                                                  # Violation Analysis
├── figures/ 
│   │
│   ├── bootstrap_distribution.pdf    : Bootstrap distribution of constant C
│   ├── uncertainty_verification.pdf  : Verification plot with violations
│   └── violation_analysis.pdf        : Multi-panel violation characterization
│
└── data/
    │
    ├── violations.json              : Metadata of principle violations
    └── robustness_report.txt        : Comprehensive statistical report
```
---

## Dependencies

The scripts require the following Python packages:

| Package     | Version |
|--------------|----------|
| Python       | 3.8+     |
| NumPy        | 1.20+    |
| SciPy        | 1.6+     |
| Matplotlib   | 3.3+     |

Exact versions used for verification and reproducibility are pinned in `requirements.txt`.

Install the pinned versions with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file included in this archive pins the package versions used to reproduce the results reported in this repository. If you prefer a conda-managed environment, use the provided `environment.yml` file.

---

### requirements.txt (pinned versions used for verification and reproducibility)

```
numpy>=1.20
scipy>=1.6
matplotlib>=3.3
```

---

## Python Environment (conda)

File: `environment.yml`

```yaml
name: QCOT
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.8
  - numpy>=1.20
  - scipy>=1.6
  - matplotlib>=3.3
```

You can activate this environment with:

```bash
conda env create -f environment.yml
conda activate QCOT
```

---

## Usage (from package root)

**Important:** The scripts are located in the `scripts/` directory. All example commands below assume you are running them from the root of this reproducibility package — that is, the directory containing `scripts/`, `figures/`, and `data/`.

### 1. Running the simulation

Run the simulation script:

```bash
python scripts/qcot_theoretical_framework_enhanced.py
```

This script produces the following files in `./figures`:

- `./figures/bootstrap_distribution.pdf`: Bootstrap distribution of constant C.
- `./figures/uncertainty_verification.pdf`: Verification plot with violations.
- `./figures/violation_analysis.pdf`: Multi-panel violation characterization.

This script produces the following files in `./data`:

- `./data/violations.json`: Metadata of principle violations.
- `./data/robustness_report.txt`: Comprehensive statistical report.
---

NOTE:
"RuntimeWarning" messages are an intrinsic and expected part of the model.
They reflect the inherent instability of the observer–world interaction
under finite perceptual resolution. This instability drives the
self-organization processes responsible for the emergence of
1/f spectral structure and Benford-like digit distributions.  

Suppressing or correcting these warnings would artificially remove
the feedback mechanism that constitutes the model’s physical content.
Therefore, the simulation must be executed in its original form,
preserving the adaptive selection of the optimal perceptual threshold
(epsilon_best) as postulated in Subjective Physics.  

---

## Citation

Vladimir Khomyakov (2025). *Empirical Verification of a Cognitive Uncertainty Principle: Bootstrapped Estimation, Confidence Intervals, and Violation Morphology*. Zenodo. https://doi.org/10.5281/zenodo.17370035

---

## Reproducibility Statement
All computational experiments described in the accompanying paper can be fully reproduced using the code and data contained in this package.
The results, figures, and numerical constants correspond exactly to those reported in the cited Zenodo release.
