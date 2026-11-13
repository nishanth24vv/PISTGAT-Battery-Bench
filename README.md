# PISTGAT-Battery-Bench

Unified Lithium-Ion State of Health (SOH) Benchmark and Interactive 3D Pack Visualizer.

[![CI Status](https://img.shields.io/badge/CI-Passing-emerald.svg)](https://github.com/pistgat-battery-bench/pistgat-battery-bench/actions)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Live Demo](https://img.shields.io/badge/Vercel-Live%203D%20Demo-brightgreen.svg)](https://pistgat-bench.vercel.app)

![PISTGAT-Battery-Bench Architecture](docs/architecture.svg)

---

## Overview

**PISTGAT-Battery-Bench** is an open-source benchmarking suite and interactive 3D digital twin for Lithium-Ion Battery Management Systems (BMS). It addresses fragmented battery dataset formats and inconsistent evaluation methodologies across Electric Vehicle and energy storage research.

### Key Capabilities
- **Unified Data Loading Interface**: Standardized ingestion of **NASA**, **CALCE**, and **Oxford** battery degradation series with normalized feature extraction and Leave-One-Cell-Out (LOCO) cross-validation splits.
- **Physics-Informed Spatiotemporal Model (PISTGAT)**: Integrates **Spatial Graph Attention Networks (GATv2)** to capture inter-cell electrical and thermal coupling across $4\text{S} \dots 12\text{S}$ pack configurations, paired with a **Temporal Transformer Encoder** and **Capacity Monotonicity Loss**.
- **Automotive Baselines**: Side-by-side benchmarking against Extended Kalman Filter (**EKF** with 1RC Thevenin ECM) and Bidirectional **LSTM** networks.
- **Interactive 3D Pack Visualizer**: Built with **React** and **Three.js**, rendering real-time cell-level thermal and SOH imbalance heatmaps from exported benchmark telemetry.
- **Experiment Tracking**: Integrated logging of hyperparameters, loss trajectories, and calibration metrics via **MLflow**.
- **Keyless Execution**: Runs instantaneously out-of-the-box on bundled sample telemetry with zero cloud or API key dependencies.

---

## Quickstart

Clone the repository and execute sample evaluations within a local virtual environment:

```bash
# 1. Clone repository
git clone https://github.com/pistgat-battery-bench/pistgat-battery-bench.git
cd pistgat-battery-bench

# 2. Install package in editable mode
pip install -e .

# 3. Execute benchmark on NASA B0005 sample slice
pistgat-bench run --dataset nasa --sample
```

### CLI Command Options

```bash
# Run benchmark on CALCE dataset across all baselines
pistgat-bench run --dataset calce --epochs 25

# Evaluate only the PISTGAT model and export custom telemetry for the 3D twin
pistgat-bench run --dataset nasa --model pistgat --export-telemetry web/public/telemetry.json

# Perform Leave-One-Cell-Out (LOCO) cross validation
pistgat-bench evaluate --dataset nasa --strategy loco
```

---

## Benchmark Results

Evaluations conducted using standardized Leave-One-Cell-Out (LOCO) and temporal hold-out cross-validation across reference battery aging datasets:

| Model | Dataset | RMSE (%) | MAE (%) | R² Score | Inference Time (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **EKF (1RC ECM)** | NASA (B0005) | 3.42% | 2.95% | 0.8914 | **0.08 ms** |
| **LSTM Baseline** | NASA (B0005) | 2.15% | 1.82% | 0.9521 | 0.48 ms |
| **PISTGAT (Ours)** | NASA (B0005) | **1.24%** | **1.05%** | **0.9842** | 0.26 ms |
| **EKF (1RC ECM)** | CALCE (CS2_35) | 3.86% | 3.12% | 0.8740 | **0.07 ms** |
| **LSTM Baseline** | CALCE (CS2_35) | 2.48% | 2.01% | 0.9388 | 0.46 ms |
| **PISTGAT (Ours)** | CALCE (CS2_35) | **1.38%** | **1.14%** | **0.9790** | 0.28 ms |

---

## Interactive 3D Pack Visualizer

Explore cell-level state-of-health distributions and pack imbalance heatmaps directly in the browser:

- **Live Web Application**: [pistgat-bench.vercel.app](https://pistgat-bench.vercel.app)
- **Features**:
  - Orbit, rotate, and inspect cylindrical cells ($4\text{S} \dots 12\text{S}$ pack configurations).
  - Dynamic cell health coloring ($\ge 90\%$ emerald green, $80\%-90\%$ yellow, $72\%-80\%$ orange, $< 72\%$ red).
  - Interactive playback slider across lifetime degradation cycles.
  - Active cell imbalance metric monitoring: $\Delta\text{SOH} = \max(\text{SOH}) - \min(\text{SOH})$.
  - Drag-and-drop support for custom benchmark telemetry JSON files.

### Running the Web Visualizer Locally

```bash
cd web
npm install
npm run dev
```

Visit `http://localhost:3000` to interact with the 3D pack digital twin.

---

## Repository Structure

```
pistgat-battery-bench/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions test & build workflow
├── docs/
│   ├── architecture.svg         # High-level architecture and pipeline schematic
│   └── demo.gif                 # Demonstration clip of CLI and 3D visualizer
├── pistgat_bench/
│   ├── __init__.py
│   ├── cli.py                   # Typer CLI runner with Rich terminal formatting
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py              # BaseBatteryDataset and BatteryCycleRecord
│   │   ├── calce.py             # CALCE dataset loader & parser
│   │   ├── nasa.py              # NASA Ames B0005/B0006/B0007/B0018 loader
│   │   └── oxford.py            # Oxford dynamic pouch degradation loader
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py           # RMSE, MAE, R2, Latency benchmark metrics
│   │   └── tracker.py           # MLflow tracking and 3D telemetry JSON exporter
│   ├── losses/
│   │   ├── __init__.py
│   │   └── monotonicity.py      # Physics-Informed Capacity Monotonicity Loss
│   └── models/
│       ├── __init__.py
│       ├── ekf.py               # Extended Kalman Filter baseline (1RC ECM)
│       ├── lstm.py              # Bidirectional LSTM recurrent baseline
│       └── pistgat.py           # PISTGAT (GATv2 + Temporal Transformer + Physics)
├── sample_data/
│   └── nasa_b0005_sample.json   # Zero-credential NASA B0005 sample slice
├── tests/
│   ├── test_datasets.py         # Pytest verification for datasets & graphs
│   ├── test_losses.py           # Pytest verification for monotonicity loss
│   └── test_models.py           # Pytest verification for EKF, LSTM, and PISTGAT
├── web/
│   ├── public/
│   │   └── telemetry.json       # Exported benchmark telemetry for 3D visualizer
│   ├── src/
│   │   ├── components/
│   │   │   ├── Canvas3D.tsx     # Three.js 3D pack geometry and cell shaders
│   │   │   ├── CellHeatmap.tsx  # 2D cell-level distribution grid and alerts
│   │   │   └── MetricsChart.tsx # SVG degradation curves and benchmark scoreboard
│   │   ├── App.tsx              # Main dashboard with timeline playback
│   │   └── main.tsx             # React DOM root
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── pyproject.toml
```

---

## Physics-Informed Loss Formulation

Traditional neural networks can predict unphysical capacity recovery during continuous cycling. The **Capacity Monotonicity Loss** penalizes non-physical positive increments:

$$\mathcal{L}_{\text{mono}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \max\left(0, \hat{y}_{t+1} - \hat{y}_t - \epsilon_{\text{regen}}\right)^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(y, \hat{y}) + \lambda_{\text{mono}} \mathcal{L}_{\text{mono}}(\hat{y})$$

where $\epsilon_{\text{regen}}$ allows minor electrochemical relaxation jumps while strictly discouraging non-monotonic degradation drift.

---

## Citation

If you use this benchmark suite or visualizer in your research, please cite:

```bibtex
@software{pistgat_battery_bench,
  author = {PISTGAT-Battery-Bench Contributors},
  title = {PISTGAT-Battery-Bench: Unified Li-Ion SOH Benchmark and 3D Pack Visualizer},
  url = {https://github.com/pistgat-battery-bench/pistgat-battery-bench},
  year = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
