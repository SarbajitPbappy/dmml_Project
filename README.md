# RetroX: Climate-Driven Dengue Outbreak Prediction (DMML)

End-to-end, production-style project that predicts weekly dengue cases **4 weeks ahead** using climate signals (temperature, rainfall, humidity, NDVI) + epidemiological momentum, and turns forecasts into **actionable 3-level alerts**.

## What you get
- **Forecasting**: train/evaluate models with time-series CV (no leakage)
- **Feature engineering**: lags (1–4w), rolling means (4/8w), autoregressive case terms
- **ERI (Epidemic Risk Index)**: interpretable climate-only risk score \([0,1]\)
- **ERS (Epidemic Risk Score)**: Normal / Elevated / High alert levels from predicted cases
- **Explainability**: SHAP-based local & global explanations (tree models)
- **API service**: FastAPI endpoints for forecast, alert, and “what-if” climate simulation
- **Dashboard**: Streamlit app for decision-makers
- **Ops**: Docker, model artifact versioning, basic drift checks

## Dataset (DengAI)
This project auto-detects DengAI files from either:
- `data/raw/dengai/` (canonical), or
- `data/` (direct Kaggle export folder)

Supported filenames include both short and Kaggle-export names:
- `dengue_features_train.csv`
- `dengue_labels_train.csv`
- `dengue_features_test.csv` (optional; for generating submissions)
- `DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv`
- `DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv`
- `DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv`

If you don’t have the dataset yet, download from Kaggle “DengAI: Predicting Disease Spread”.

## Quickstart
Create an environment and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Train and evaluate (default: RandomForest + time-series CV):

```bash
python -m retrox.cli.train --city sj --horizon 4
python -m retrox.cli.evaluate --city sj
```

Run the API:

```bash
uvicorn retrox.api.main:app --reload
```

Run the dashboard:

```bash
streamlit run retrox/dashboard/app.py
```

## Project layout
- `retrox/` - python package
  - `data/` - loaders + schema utilities
  - `features/` - lag/rolling/AR features + ERI
  - `models/` - training, evaluation, persistence
  - `explain/` - SHAP reports
  - `api/` - FastAPI service
  - `dashboard/` - Streamlit app
- `artifacts/` - trained models, metrics, reports (created by scripts)
- `data/` - local data (not committed)

## Notes / defaults
- **Forecast horizon** is configurable; PPT target is **4 weeks**.
- **ERS thresholds** default to the PPT values: Normal \< 39, Elevated 39–71, High \> 71 (configurable).
- **ERI weights** default to the PPT: Temp 0.35, Humidity 0.30, Rainfall 0.25, NDVI 0.10 (configurable).

