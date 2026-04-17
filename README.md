# RetroX: Dengue Forecasting + Universal Data Intelligence

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![API](https://img.shields.io/badge/API-FastAPI-009688)
![ML](https://img.shields.io/badge/ML-scikit--learn-F7931E)
![Tuning](https://img.shields.io/badge/Tuning-Optuna-4B8BBE)

RetroX is an end-to-end time-series intelligence platform that combines:

- dengue early-warning forecasting (DengAI-compatible),
- universal tabular forecasting for uploaded datasets,
- automated EDA and preprocessing,
- model comparison + tuning,
- SHAP-based explanation.

It is designed to run as a Streamlit app, CLI toolkit, and FastAPI service.

## Features

- **Forecast + Alerting dashboard** with city/horizon controls and dynamic target mapping.
- **Leakage-safe feature engineering** with lag, rolling, EWM, cyclical week, and autoregressive target features.
- **Environmental Risk Index (ERI)** from weighted normalized climate signals.
- **Epidemic Risk Signal (ERS)** classification (`normal`, `elevated`, `high`) from predicted case count.
- **Dual training modes**:
  - full candidate search (HistGBM, GBM, RF, ExtraTrees, Ridge),
  - fast mode for custom uploads (HistGBM + RF, minimal tuning).
- **Data Lab pipeline**: smart EDA -> preprocessing controls -> t-SNE/PCA -> AutoML -> suggestions/report export.
- **Model explainability** via SHAP waterfall on latest prediction.
- **Artifact-based deployment** using `joblib` model + JSON metadata.

## Tech Stack

- Python, Pandas, NumPy
- scikit-learn, Optuna, SHAP
- Streamlit, Plotly
- FastAPI + Pydantic

## Quick Start

### 1) Install

```bash
git clone <your-repository-url>
cd "dmml project"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use the project Makefile:

```bash
make venv
make install
make verify
```

### 2) Run Dashboard

```bash
streamlit run retrox/dashboard/app.py
```

Then open `http://localhost:8501`.

### 3) Optional CLI Usage

Train:

```bash
python -m retrox.cli.train --city sj --horizon 4
```

Drift report:

```bash
python -m retrox.cli.drift --city sj --recent-weeks 52
```

Explainability artifact export:

```bash
python -m retrox.cli.explain --city sj --horizon 4
```

### 4) Optional API Usage

Run your ASGI server against `retrox.api.main:app` and use:

- `GET /health`
- `POST /forecast`
- `POST /whatif`

## High-Level Workflow

1. Load built-in DengAI data or upload your dataset.
2. Configure target/date mapping in dashboard.
3. Generate features and load compatible model artifact.
4. Forecast future value for selected horizon.
5. Show risk outputs:
   - dengue mode: ERI + ERS,
   - generic mode: anomaly level + trend velocity.
6. Explain prediction with SHAP and backtesting views.
7. Retrain if target/horizon/features are incompatible with saved artifact.

## Repository Structure

```text
retrox/
  api/                # FastAPI endpoints and request/response schemas
  cli/                # Train/evaluate/drift/explain command-line tools
  dashboard/          # Streamlit app and Data Lab pipeline
  data/               # DengAI loaders and synthetic demo generator
  explain/            # Explainability report generators
  features/           # Feature engineering, ERI, ERS
  models/             # Training, inference, artifact registry
  ops/                # Drift metrics (PSI)
DOCUMENTATION.md      # Full implementation-level technical documentation
README.md             # Project-facing quick guide
```

## Artifacts and Outputs

Trained model artifacts are saved under:

`artifacts/models/city=<city>/h=<horizon>/`

Typical files:

- `random_forest.joblib`
- `random_forest.meta.json`

Data Lab and explainability stages can also export analysis artifacts (CSV/markdown) depending on workflow.

## Documentation

For full module/function-level details, formulas, and internal procedure, read:

- [DOCUMENTATION.md](DOCUMENTATION.md)
