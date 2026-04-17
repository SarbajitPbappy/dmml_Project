# 🦟 RetroX: Universal AI Forecasting Engine

![Project Status](https://img.shields.io/badge/Status-Production--Ready-success)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E)
![Optuna](https://img.shields.io/badge/AutoML-Optuna-4B8BBE)

**RetroX** is a production-ready, universal time-series forecasting engine. Originally engineered to conquer the DengAI epidemiological forecasting challenge, the platform has fundamentally evolved to support **any arbitrary tabular dataset**. 

RetroX bypasses traditional notebook workflows, providing an end-to-end laboratory that ingests data, performs Automated Exploratory Data Analysis (EDA), synthesizes deep chronological features, optimizes hyper-parameters, and delivers SHAP-driven predictive intelligence—all through a beautifully designed, responsive user interface.

## ✨ Core Features

* **Universal Feature Engineering**: Automatically extracts chronological lags (1, 2, 4... 52 weeks) and rolling statistics exclusively for numeric targets, preventing catastrophic scaling errors on temporal/string columns.
* **Dual-Mode AutoML**:
  * **Exhaustive Mode**: Deep search execution testing Gradient Boosters, Random Forests, Extra Trees, and Ridge Regression over 10 optimization cycles utilizing TimeSeries Cross Validation.
  * **Fast Universal Mode**: An ultra-fast heuristic engaged for custom uploads, shrinking optimization to the most reliable algorithms (HistGBM & RandomForest) and generating real-time models in seconds.
* **Intelligent Data Lab**: Instantly processes unstructured CSVs to visualize distribution histograms, t-SNE dimensionality reduction, variance thresholds, and correlation matrices. 
* **Dynamic UI Extensibility**: Instead of relying on static features, the dashboard intelligently shifts context. For Dengue tasks, it flags the Environmental Risk Index (ERI); for Custom Tasks (e.g., Finance/Operations), it automatically computes Anomaly Z-Scores and Trend Velocities.
* **SHAP Explainability**: De-black-boxes the algorithms via waterfall plots, highlighting the exact vector influence (positive/negative) of individual features per forecast.

## 🚀 Getting Started

### Prerequisites

* Python 3.11 or higher
* `make` (optional, for utilizing the Makefile)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd "dmml project"
   ```

2. **Set up the virtual environment & install dependencies:**
   ```bash
   make venv
   make install
   ```

3. **Verify the environment**:
   ```bash
   make verify
   ```

### Running the Platform

To launch the RetroX dashboard locally:
```bash
make dashboard
```
The application will be served at `http://localhost:8501`.

## 📁 Repository Structure

```text
├── retrox/
│   ├── dashboard/         # Streamlit UI layers
│   │   ├── app.py         # Main entry point and Forecast Hub
│   │   └── lab_pipeline.py# Automated Data Lab & EDA logic
│   ├── features/          # Deep logic for temporal engineering
│   │   ├── engineering.py # Builds massive lag & rolling data matrices
│   │   └── eri.py         # Dengue-specific Environmental Risk Index formulation
│   └── models/            # Intelligence Pipeline
│       ├── training.py    # Optuna HPO, Fast/Deep Model logic
│       └── inference.py   # Prediction matrix, Artifact retrieval, SHAP execution
├── artifacts/             # Persisted joblib model artifacts
├── Makefile               # Streamlined command sequences
├── capture_screenshots.py # Playwright automated visual reporting
└── generate_final_report.py # Docx Lab Generation script
```

## 🧠 Documentation

For a comprehensive technical deep-dive into the architectural decisions, pipeline workflows, error handling methodologies, and specific hyperparameter parameters, please refer to the [DOCUMENTATION.md](DOCUMENTATION.md) included in this repository.
