# The Temporal Efficiency Paradox

## Fast Sensing, Slow Understanding: Neural Evidence for Hierarchical Temporal Dynamics

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CCN 2026](https://img.shields.io/badge/Conference-CCN%202026-green.svg)](https://ccneuro.org/)

---

## Overview

This project investigates the **temporal dynamics of brain activity during naturalistic movie watching**, examining the relationship between processing speed and encoding accuracy across cortical networks. We reveal a fundamental paradox: what appears "inefficient" from a speed perspective is actually optimal for deep understanding.

### Key Findings

| Network | TDS (Speed) | DCS (Stability) | TWG (Long-window Benefit) |
|---------|-------------|-----------------|---------------------------|
| Frontoparietal | 1.24 (Fastest) | 0.97 | +0.035 |
| Dorsal Attention | 1.14 | 0.93 | +0.020 |
| Default Mode | 0.84 | 0.96 | **+0.059** (Highest) |
| Ventral Attention | 0.67 (Slowest) | 0.92 | +0.048 |

**The Paradox**: Default Mode Network (DMN) processes information slowly but benefits most from extended temporal integration—revealing that "slow" is not inefficient, but optimized for deep semantic understanding.

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

### Redefining Efficiency

> **Traditional view**: Efficiency = Speed. Faster is better.
> 
> **New view**: Efficiency = Goal-alignment. Fast for reaction, slow for understanding.

### The Dual-Timescale Architecture

The brain implements a **hierarchical temporal architecture**:
1. **Fast pathway** (Sensory networks): Optimized for rapid stimulus detection
2. **Slow pathway** (Integration networks/DMN): Optimized for deep semantic understanding

### Implications for AI Design

1. Don't optimize all components for speed—some modules need to be "slow"
2. Implement **hierarchical temporal architectures**: fast perception + slow reasoning
3. Allow "slow" pathways to exist—deep understanding requires temporal accumulation

---

## Citation

If you use this code or findings, please cite:

```bibtex
@inproceedings{temporal_efficiency_paradox_2026,
  title={The Temporal Efficiency Paradox: Fast Brains React, Slow Brains Understand},
  author={[Authors]},
  booktitle={Cognitive Computational Neuroscience (CCN)},
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

*Last updated: January 2026*
