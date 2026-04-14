# The Temporal Efficiency Paradox

## Fast Brains React, Slow Brains Understand

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CogSci 2026](https://img.shields.io/badge/Conference-CogSci%202026-green.svg)](https://cognitivesciencesociety.org/)

---

## Overview

This project investigates the **temporal dynamics of brain activity during naturalistic movie watching**, examining the relationship between processing speed and encoding accuracy across cortical networks.

### Key Findings

| Network | TDS | DCS | TWG | Ridge *r* | LSTM *r* | LSTM Gain |
|---------|-----|-----|-----|-----------|----------|-----------|
| Frontoparietal | 1.20 (Fastest) | 0.30 | -0.015 | 0.277 | 0.408 | +47.5% |
| Limbic | 0.95 | 0.46 | -0.024 | 0.305 | 0.456 | +49.3% |
| Dorsal Attention | 0.93 | 0.51 | -0.026 | 0.285 | 0.425 | +49.5% |
| Somatomotor | 0.89 | 0.45 | -0.029 | 0.327 | 0.514 | +57.4% |
| Visual | 0.82 | 0.57 | -0.028 | 0.317 | 0.492 | +55.4% |
| Default Mode | 0.78 | 0.38 | -0.021 | 0.386 | 0.582 | +50.8% |
| Ventral Attention | 0.64 (Slowest) | 0.51 | -0.029 | 0.423 | 0.632 | +49.3% |

**The Paradox Resolved**: We hypothesized that slower networks would benefit more from temporal integration, but this was not supported—temporal averaging (TWG) impaired encoding across *all* networks. However, LSTM models that preserve sequential dependencies improved encoding by 47-57% uniformly, even over ridge regression with concatenated temporal features. The key factor is not intrinsic dynamics speed, but whether temporal structure is properly modeled.

---

## Project Structure

```
project_2/
├── run_analysis.py              # Unified analysis pipeline
├── 01_temporal_analysis.py      # Temporal dynamics analysis (TDS)
├── 02_dynamic_fc.py             # Dynamic functional connectivity (DCS)
├── 03_encoding.py               # Multi-timescale encoding (TWG)
├── 04_statistics.py             # Statistical testing
├── 05_generate_figures.py       # Figure generation
├── 05_individual.py             # Individual differences analysis
├── 06_lstm_comparison.py        # LSTM vs Ridge comparison
├── 07_hub_analysis.py           # Hub centrality analysis
├── 08_transformer.py            # Transformer attention analysis
├── PROJECT_COMPREHENSIVE_DOCUMENTATION.md  # Full manuscript
├── PROJECT_DESIGN.md            # Research design document
├── FIGURE_DOCUMENTATION.md      # Figure descriptions
├── requirements.txt             # Python dependencies
└── runs/                        # Analysis results
    └── TIMESTAMP/
        ├── temporal_spectral/   # TDS results
        ├── dynamic_fc/          # DCS results
        ├── multiscale_encoding/ # TWG results
        ├── figures/             # Generated figures (PNG + SVG)
        └── config.json          # Run configuration
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for deep learning models)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/temporal-efficiency-paradox.git
cd temporal-efficiency-paradox

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Requirements

This project uses the **Algonauts 2025 Challenge** dataset with the **CNeuroMod Friends** fMRI data:

- 4 subjects (sub-01, sub-02, sub-03, sub-05)
- Schaefer 2018 Atlas (1000 parcels, 7 networks)
- Friends TV series naturalistic viewing paradigm

Place the data in `../data/` relative to the project directory.

---

## Quick Start

```bash
# Quick analysis (4 episodes, ~5 minutes)
python run_analysis.py

# Full analysis (all data, ~60 hours)
python run_analysis.py --mode full

# Custom episodes
python run_analysis.py --max_episodes 20

# Skip AI models (faster)
python run_analysis.py --skip_ai

# Generate figures only (from existing results)
python run_analysis.py --figures_only --run_dir runs/TIMESTAMP
```

---

## Analysis Pipeline

### Step 1: Temporal Dynamics Analysis (TDS)

Computes power spectral density for each brain region and calculates the ratio of high-frequency to low-frequency power.

**Metric**: TDS = P_HF / P_LF (High-frequency: 0.07-0.25 Hz, Low-frequency: 0.01-0.07 Hz)

### Step 2: Dynamic Functional Connectivity (DCS)

Analyzes the temporal stability of functional connections using sliding-window correlation.

**Metric**: DCS = 1 - σ/σ_max (Stability of within-network connectivity over time)

### Step 3: Multi-Timescale Encoding (TWG)

Tests whether networks benefit from extended temporal integration using ridge regression encoding models.

**Metric**: TWG = r_60TR - r_1TR (Encoding improvement with longer temporal context)

### Step 4: Statistical Testing

Performs hypothesis testing for network differences (permutation tests, effect sizes, FDR correction).

### Step 5: Individual Differences

Analyzes cross-subject variability and reliability (ICC analysis).

### Step 6-8: AI Model Analysis (Optional)

- **LSTM vs Ridge**: Tests if recurrent models improve DMN encoding more than Visual
- **Hub Centrality**: Analyzes network topology to explain why DMN needs slow processing
- **Transformer Attention**: Visualizes temporal attention patterns across networks

---

## Key Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **TDS** (Temporal Dynamics Speed) | Ratio of high-frequency to low-frequency power | TDS > 1: Fast dynamics, TDS < 1: Slow dynamics |
| **DCS** (Dynamic Connectivity Stability) | Temporal consistency of functional connections | DCS → 1: Stable, DCS → 0: Variable |
| **TWG** (Temporal Window Gain) | Benefit of extended temporal integration | TWG > 0: Benefits from long context |

---

## Generated Figures

| Figure | Description | Research Question |
|--------|-------------|-------------------|
| Fig 1 | Temporal Spectrum Hierarchy | Q1: Network speed differences |
| Fig 2 | Dynamic Connectivity | Q2: Connection stability |
| Fig 3 | Multi-scale Encoding | Q3: Temporal integration benefits |
| Fig 4 | Network Summary Radar | Overview of all metrics |
| Fig 5 | LSTM vs Ridge Comparison | AI model analysis |
| Fig 6 | Hub Centrality | Network topology |
| Fig 7 | Transformer Attention | Temporal attention patterns |
| Fig 8 | Brain Glass Overview | Spatial distribution of TDS |

---

## Theoretical Contribution

### The Paradox Resolved: Two-Level Distinction

1. **Temporal averaging destroys information** — collapsing features across time impairs encoding (negative TWG for all networks)
2. **Concatenation preserves information but not structure** — ridge regression with concatenated features still underperforms LSTM by 47-57%
3. **Sequential modeling is key** — LSTM learns temporal dependencies, benefiting all networks uniformly regardless of intrinsic dynamics speed

### Implications

- **For neuroscience**: Cortical temporal hierarchy is real (RQ1), but temporal integration benefits all networks equally when properly modeled, suggesting a parallel multi-timescale architecture rather than a simple speed gradient
- **For encoding models**: Preserve temporal dependencies rather than aggregate features across time; the ~50% LSTM improvement demonstrates temporal structure accounts for substantial neural response variance
- **For AI design**: Recurrent, order-sensitive architectures capture temporal structure that static or averaging approaches miss

---

## Citation

If you use this code or findings, please cite:

```bibtex
@inproceedings{temporal_efficiency_paradox_2026,
  title={The Temporal Efficiency Paradox: Fast Brains React, Slow Brains Understand},
  author={[Authors]},
  booktitle={Proceedings of the Annual Conference of the Cognitive Science Society (CogSci)},
  year={2026}
}
```

Please also cite the Algonauts 2025 Challenge and the CNeuroMod dataset.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Algonauts 2025 Challenge** for providing the dataset framework
- **CNeuroMod Project** for the Friends fMRI dataset
- **Schaefer et al. (2018)** for the cortical parcellation atlas

---

*Last updated: April 2026*
