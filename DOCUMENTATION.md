# RetroX Technical Documentation (Code-Aligned)

This document describes the full architecture and behavior of the RetroX project under `retrox/`: data flow, feature engineering, model training, forecasting, explainability, Data Lab workflow, API, CLI, persistence, and important implementation caveats.

---

## 1) Project Overview

RetroX is a time-series forecasting platform centered on dengue early-warning, with support for arbitrary uploaded tabular data in the Streamlit UI.

Main capabilities:

1. DengAI dataset loading and synthetic demo-data generation.
2. Leakage-safe time-series feature engineering.
3. Multi-model training with Optuna tuning and time-series CV.
4. Forecasting with compatibility checks against saved artifacts.
5. SHAP-based local explanation for latest forecast.
6. A full Data Lab pipeline (EDA -> preprocessing -> t-SNE/PCA -> AutoML -> suggestions).
7. FastAPI forecast endpoints.
8. CLI tools for train, drift, and explain workflows.

---

## 2) End-to-End Working Procedure

### 2.1 Forecast & Alerting flow (Dashboard tab 1)

1. User picks `city` (`sj`/`iq`) and `horizon` (1-12 weeks) in sidebar.
2. Data source:
   - Built-in DengAI (last 60 weeks subset), or
   - Uploaded CSV/Excel data.
3. User maps:
   - Numeric target column (`y_col`), and
   - Optional time/date column.
4. App loads existing artifact from:
   - `artifacts/models/city=<city>/h=<horizon>/random_forest.joblib`
   - `artifacts/models/city=<city>/h=<horizon>/random_forest.meta.json`
5. Compatibility checks:
   - Target in artifact metadata must match requested target/horizon (with legacy fallback),
   - Required feature columns must exist after current feature generation.
6. If compatible:
   - Build last feature row,
   - Run prediction,
   - Compute risk output (dengue mode: ERS + ERI, universal mode: anomaly + velocity),
   - Render SHAP waterfall explanation,
   - Render historical backtesting chart and metrics.
7. If missing/incompatible:
   - UI prompts targeted retraining via `train_model(...)`,
   - On success reruns and serves forecast.

### 2.2 Data Lab flow (Dashboard tab 2)

1. Upload any tabular dataset (or load DengAI).
2. Stage 1: Smart EDA.
3. Stage 2: Preprocessing controls and transformation.
4. Optional export of processed data (CSV/Parquet).
5. Stage 3: t-SNE visualization (with PCA pre-reduction when needed).
6. Stage 4: AutoML for regression/classification with cross-validation.
7. Stage 5: Research suggestions + downloadable markdown report.

---

## 3) Configuration and Core Utilities

### 3.1 `retrox/config.py`

`Settings` (`pydantic-settings`) reads env vars with `RETROX_` prefix.

Key config:
- `project_root`
- `data_dir`, `artifacts_dir` (optional overrides)
- ERI weights:
  - `eri_w_temperature = 0.35`
  - `eri_w_humidity = 0.30`
  - `eri_w_rainfall = 0.25`
  - `eri_w_ndvi = 0.10`
- ERS thresholds:
  - `ers_normal_max = 38`
  - `ers_elevated_max = 71`
- defaults:
  - `default_city = "sj"`
  - `default_horizon_weeks = 4`

Path helpers:
- `resolved_data_dir()`
- `resolved_artifacts_dir()`

### 3.2 `retrox/logging_utils.py`

`get_logger(name)` returns a standard library logger with stream handler and format:
`timestamp | level | logger_name | message`.

---

## 4) Data Layer (`retrox/data`)

### 4.1 `dengai.py`

#### Types
- `City = Literal["sj", "iq"]`
- `DengAIPaths(root: Path)` with properties:
  - `features_train`
  - `labels_train`
  - `features_test`
  Each property supports two filename variants and chooses first existing file.

#### File location behavior
- `_default_dengai_root()`:
  - prefers `<data_dir>/raw/dengai` if exists, else `<data_dir>`.
- `dengai_files_available(root=None) -> bool`:
  - true if training features and labels exist.

#### Loaders
- `load_dengai_train(city, root=None) -> DataFrame`
  - reads train features + labels,
  - merges on `city/year/weekofyear`,
  - filters by city,
  - parses `week_start_date` if present,
  - sorts by `year/weekofyear`.
- `load_dengai_test(city, root=None) -> DataFrame`
  - reads test features only,
  - filters/sorts similarly.

#### Synthetic dataset generator
- `write_demo_dataset(root=None, years=18, holdout_weeks=16, seed=42) -> DengAIPaths`

What it simulates:
- seasonality (sinusoidal waves),
- 2-4 week lag effects in climate vectors,
- autoregressive outbreak momentum (`prev_cases_1`, `prev_cases_2`),
- city-specific profiles (`sj`, `iq`).

Outputs written:
- train features CSV,
- train labels CSV,
- test features CSV.

---

## 5) Feature Engineering and Risk Calculations (`retrox/features`)

### 5.1 `engineering.py`

#### Constants and data structures
- `ID_COLUMNS = ["city", "year", "weekofyear", "week_start_date"]`
- `SupervisedFrame`:
  - `frame`, `feature_columns`, `id_columns`, `target_column`
- `FeatureParams`:
  - `lags=(1,2,4,8,12,26,52)`
  - `rolling_windows=(4,8,12,26)`
  - `include_autoregressive=True`
  - `target_column="total_cases"`
  - `date_column="week_start_date"`

#### Signal detection and prep
- `infer_signal_columns(df)` maps columns into signal groups by name matching:
  - temperature (`temp`, `temperature`, `dew_point`)
  - humidity (`humidity`)
  - rainfall (`precip`, `rain`)
  - ndvi (`ndvi`, `vegetation`)
- `_dedupe(columns)` preserves order.
- `_fill_numeric(df)` performs:
  - interpolation (`limit_direction="both"`),
  - then `ffill` and `bfill`.
- `_mean_signal(df, columns, fallback)` averages grouped signal columns.

#### Main feature generator
- `make_feature_frame(df, params=None) -> DataFrame`

Processing steps:
1. Sorts by date column if present, else by year/week, else year.
2. Fills numeric missing values.
3. If dengue-like signals detected:
   - creates `signal_temperature`, `signal_humidity`, `signal_rainfall`, `signal_ndvi`,
   - computes `eri`.
4. Creates cyclical week features:
   - `week_sin = sin(2*pi*week/52)`
   - `week_cos = cos(2*pi*week/52)`
5. For numeric base features (excluding IDs/target and a small ignore-set):
   - lag features: `col_lag_<k>`
   - rolling mean/std: `col_roll_mean_<w>`, `col_roll_std_<w>`
   - exponential moving average: `col_ewm_mean_<w>`
6. For numeric target column:
   - target lags: `target_lag_<k>`
   - target rolling mean: `target_roll_mean_<w>`
   - target momentum: `target_diff_<w>`
   - special dengue feature:
     - `outbreak_momentum = total_cases_roll_mean_4 * (1 + eri)` when applicable.

#### Supervised frame builder
- `build_supervised_frame(df, horizon=4, horizon_weeks=None, params=None, target_column=None) -> SupervisedFrame`

Target construction:
- Creates shifted prediction target:
  - `future_target_column = f"target_{target_col}_t_plus_{horizon}"`
  - value = `target_col.shift(-horizon)`

Row validity:
- keeps rows where future target is not null and feature non-null ratio >= 0.7.

Other helpers:
- `numeric_feature_columns(df)`
- `clean_numeric_frame(df, cols)` with `pd.to_numeric(..., errors="coerce")`
- `last_complete_feature_row(df, horizon_weeks, params=None)` -> last engineered row.

### 5.2 `eri.py` (Environmental Risk Index)

#### Types
- `ERIWeights(temperature, humidity, rainfall, ndvi)` from config defaults.

#### Normalization
- `_minmax(series)`:
  - min-max scales to `[0,1]`,
  - returns zeros if non-finite or constant series.

#### Main ERI formula
- `compute_eri(df, temperature_col, humidity_col, rainfall_col, ndvi_col, weights=None, out_col="eri")`

Formula:

`eri = (w_t*temp_n + w_h*hum_n + w_r*rain_n + w_n*ndvi_n) / (w_t+w_h+w_r+w_n)`

Final clamp: `eri.clip(0, 1)`.

### 5.3 `ers.py` (Epidemic Risk Signal)

- `classify_ers(pred_cases)`:
  - returns `"normal"` if non-finite or `<= 38`
  - `"elevated"` if `39..71`
  - `"high"` if `> 71`

---

## 6) Model Training, Registry, and Inference (`retrox/models`)

### 6.1 `training.py`

#### Candidate models
- `HistGradientBoostingRegressor`
- `GradientBoostingRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `Ridge`

#### Preprocessing pipeline
- `_build_preprocessor(feature_cols)`:
  - `FunctionTransformer` to force numeric conversion,
  - `SimpleImputer(strategy="median")`,
  - wrapped in `ColumnTransformer`.

#### Hyperparameter tuning
- `_tune_model(name, X, y, feature_cols, n_trials, random_state)`
  - Optuna objective minimizes MAE using `TimeSeriesSplit(n_splits=3)`.
  - Parameter spaces vary by model family.

#### Evaluation
- `_cv_scores_for_estimator(..., n_splits=5)` computes:
  - `mae`, `rmse`, `r2`, `splits`.
- `compare_models(...)` runs CV across base candidates.

#### Training and artifact creation
- `train_model(df, city, horizon_weeks, params, random_state=42, artifact_name="random_forest", extra_meta=None, target_column=None, n_tuning_trials=3, fast_mode=False)`

Behavior:
- Builds supervised frame.
- Chooses candidate list:
  - full list in normal mode,
  - only `HistGradientBoosting` + `RandomForest` when `fast_mode=True` and sets `n_tuning_trials=1`.
- Tunes each candidate -> CV -> picks lowest MAE.
- Fits best pipeline on full data.
- Stores metadata:
  - city/horizon/model type,
  - feature params and columns,
  - target column,
  - CV metrics for all models,
  - best model id,
  - `background_sample` (up to 100 rows) for SHAP background.
- Saves artifact via registry.

### 6.2 `registry.py`

- `default_model_dir(city, horizon_weeks)`:
  - `<artifacts_dir>/models/city=<city>/h=<horizon_weeks>`
- `ModelArtifact(model_path, meta_path)`:
  - `load_model()` via `joblib`,
  - `load_meta()` via JSON.
- `save_artifact(model, meta, out_dir, name)`:
  - ensures dir exists,
  - writes `<name>.joblib` and `<name>.meta.json`,
  - auto-injects `created_at` (UTC) if missing.

### 6.3 `inference.py`

- `load_latest_artifact(model_dir, name="random_forest")`
  - validates both model and meta files exist.

- `predict_next(history_df, artifact, horizon_weeks, params=None) -> float`
  - computes latest feature row,
  - reads expected feature columns from metadata,
  - checks missing required columns and raises explicit error if mismatch,
  - runs model `.predict`,
  - clamps negative predictions to `0`.

- `explain_prediction(history_df, artifact, horizon_weeks, params=None) -> dict`
  - gets latest feature row and model pipeline steps (`pre`, `model`),
  - transforms current sample + background sample,
  - runs `shap.Explainer(estimator, bg_tf)` then `explainer(X_tf)`,
  - returns shap values, feature names, base value,
  - returns `{"error": ...}` on failure.

---

## 7) Dashboard App (`retrox/dashboard/app.py`)

### 7.1 UI framework
- Streamlit app with dark premium CSS theme and two tabs:
  1. `Forecast & Alerting`
  2. `Data Lab`

Sidebar controls:
- city selector (`sj`/`iq`)
- forecast horizon slider (1-12)

### 7.2 Data ingestion helpers
- `_read_file(f)` supports CSV, Excel, JSON, Parquet.
- `_artifact_safe(...)` wraps artifact loading with error handling.
- `_ensure_demo()` auto-generates demo DengAI dataset if missing.
- `_dark_chart(fig)` applies theme settings.

### 7.3 Forecast tab behavior

Modes:
- built-in DengAI data,
- generic file upload.

Universal mapping:
- target numeric column selection,
- optional date/time column selection.

Inference gatekeeping:
- model target/horizon compatibility checks,
- feature-column compatibility checks.

Forecast output cards:
- always: predicted target in +horizon weeks.
- dengue mode (`total_cases` or built-in mode):
  - alert level from ERS,
  - ERI risk index.
- universal mode:
  - anomaly level from z-score vs last 20 historical points,
  - trend velocity:
    - `(pred - last_observed) / horizon`.

SHAP explainability:
- renders waterfall plot for latest prediction.

Historical backtesting block:
- rebuilds supervised frame,
- predicts on historical feature matrix,
- displays actual vs fitted lines + future forecast marker,
- computes backtest MAE and R2 on lead-shifted target.

Training fallback:
- if no compatible artifact, offers button that calls `train_model(...)`,
- uses `fast_mode=True` automatically for non-built-in uploads.

---

## 8) Data Lab Analytics Pipeline (`retrox/dashboard/lab_pipeline.py`)

### 8.1 Stage 1 - `render_smart_eda(df)`

Includes:
- row/column/type/missing/duplicate metrics,
- schema table,
- numeric summary with skewness, kurtosis, coefficient of variation, outlier count,
- outlier table via 3-sigma rule,
- missingness bar chart,
- Pearson correlation heatmap + high-correlation pair listing,
- interactive histogram/box/violin explorer,
- scatter matrix (up to configured numeric columns),
- categorical value-count chart,
- time-series plot when date-like column detected.

### 8.2 Stage 2 - `render_preprocessing_controls(df) -> DataFrame`

User-configurable transforms:
- imputation strategy: median/mean/most_frequent/constant,
- scaling: none/standard/minmax,
- outlier winsorization (1st-99th percentile),
- one-hot encoding for categorical columns,
- drop near-zero variance numeric columns with threshold slider,
- drop columns with >50% missing.

Also standardizes column names, removes duplicate rows, and shows shape delta.

### 8.3 Stage 3 - `render_tsne(processed_df, raw_df)`

Pipeline:
- requires >=2 numeric columns,
- optional row sampling up to `_TSNE_MAX_ROWS`,
- optional PCA to 50 components before t-SNE,
- supports 2D or 3D embedding,
- user controls perplexity and iterations,
- optional color by raw column,
- renders Plotly scatter and KL-divergence caption.

### 8.4 Stage 4 - AutoML

Task detection:
- `_detect_task(series)` decides regression/classification from dtype and cardinality.

CV helpers:
- `_cv_regression` uses `TimeSeriesSplit` and returns MAE/RMSE/R2.
- `_cv_classification` uses `StratifiedKFold` and returns Accuracy/F1/AUC-ROC.

Main function:
- `render_automl(processed_df, raw_df) -> dict`
  - chooses target column (with hint-based defaults),
  - trains/evaluates model set,
  - renders leaderboard + metric visuals,
  - renders diagnostics:
    - regression: residuals, actual-vs-predicted, trend lines
    - classification: confusion matrix, classification report, ROC (binary)
  - renders feature-importance charts for tree-based models.

Models used:
- Regression:
  - HistGradientBoostingRegressor
  - GradientBoostingRegressor
  - RandomForestRegressor
  - KNeighborsRegressor
  - Ridge
- Classification:
  - HistGradientBoostingClassifier
  - GradientBoostingClassifier
  - RandomForestClassifier
  - LogisticRegression
  - KNeighborsClassifier

### 8.5 Stage 5 - `render_suggestions(...)`

Generates:
- strategic recommendations,
- technical tuning suggestions,
- downloadable markdown report containing profile + leaderboard + roadmap.

---

## 9) API Layer (`retrox/api`)

### 9.1 Schemas (`schemas.py`)

- `WeeklyRecord`: optional fields for weekly metadata and climate variables.
- `ForecastRequest`:
  - `city`
  - `horizon_weeks` (1-12)
  - `history` with min length 8
- `ForecastResponse`:
  - city, horizon, predicted_cases, ers_level, eri_latest
- `WhatIfRequest` extends `ForecastRequest` with `override` record.

### 9.2 FastAPI app (`main.py`)

Endpoints:
- `GET /health` -> `{"status":"ok"}`
- `POST /forecast`
  - converts request history to DataFrame,
  - loads artifact by city+horizon,
  - predicts with `predict_next`,
  - computes latest ERI if present,
  - returns forecast response.
- `POST /whatif`
  - applies `override` values to last history record,
  - then runs same forecast flow.

---

## 10) Explainability and Drift Ops

### 10.1 `retrox/explain/shap_report.py`

- `generate_shap_report(city, horizon_weeks, model_name="random_forest", out_dir=None, max_rows=800)`

Despite filename, implementation uses **permutation importance** on transformed features:
- loads trained artifact + metadata,
- rebuilds supervised frame,
- transforms with pipeline preprocessor,
- computes `permutation_importance` using neg MAE,
- writes:
  - `shap_importance.csv`
  - `top_features.md`
- returns output paths.

### 10.2 `retrox/ops/drift.py`

- `psi(expected, actual, bins=10)` computes Population Stability Index.
- `drift_report(reference_df, recent_df, params=None)`
  - rebuilds feature frames,
  - computes PSI for selected metrics if present.

---

## 11) CLI Commands (`retrox/cli`)

- `train.py`
  - loads DengAI train by city,
  - trains model for horizon.

- `evaluate.py`
  - intended to run time-series CV and print metrics JSON.

- `drift.py`
  - compares historical reference vs recent window with PSI report.

- `explain.py`
  - generates explainability artifacts from trained model.

---

## 12) Artifact and Directory Conventions

Model artifacts are stored at:

`<artifacts_dir>/models/city=<city>/h=<horizon>/`

Typical files:
- `random_forest.joblib`
- `random_forest.meta.json`

Metadata includes:
- feature parameters and feature column list,
- target column name,
- best model and CV summaries,
- background sample used for SHAP fallback.

---

## 13) Practical Notes and Caveats

1. Forecasting is strict about feature compatibility; changing target/date schema usually requires retraining.
2. `make_feature_frame` currently creates `signal_temperature/signal_humidity/signal_rainfall/signal_ndvi`; drift code expects `temp_signal/humidity_signal/rain_signal/ndvi_signal`, so only matching columns are reported.
3. `explain/shap_report.py` currently outputs markdown in key named `summary_png`; output is not a PNG.
4. `evaluate.py` imports `time_series_cv_scores`; ensure this function exists in `training.py` or update CLI to use available evaluation function.

---

## 14) Quick Runbook

### 14.1 Streamlit app
- Run dashboard entrypoint:
  - `streamlit run retrox/dashboard/app.py`

### 14.2 Train CLI
- `python -m retrox.cli.train --city sj --horizon 4`

### 14.3 Drift CLI
- `python -m retrox.cli.drift --city sj --recent-weeks 52`

### 14.4 Explain CLI
- `python -m retrox.cli.explain --city sj --horizon 4`

### 14.5 API
- Start FastAPI app with your ASGI server and call:
  - `GET /health`
  - `POST /forecast`
  - `POST /whatif`

---

## 15) Module Index

- `retrox/__init__.py` - package version.
- `retrox/config.py` - runtime settings.
- `retrox/logging_utils.py` - logger utility.
- `retrox/data/dengai.py` - DengAI IO + synthetic generation.
- `retrox/features/engineering.py` - feature pipeline and supervised frame.
- `retrox/features/eri.py` - environmental risk index.
- `retrox/features/ers.py` - alert-level classification.
- `retrox/models/training.py` - model tuning/training/comparison.
- `retrox/models/registry.py` - artifact persistence.
- `retrox/models/inference.py` - prediction and SHAP explanation.
- `retrox/dashboard/app.py` - Streamlit UI and workflow orchestration.
- `retrox/dashboard/lab_pipeline.py` - full Data Lab analytics pipeline.
- `retrox/api/schemas.py` - API Pydantic models.
- `retrox/api/main.py` - FastAPI app.
- `retrox/ops/drift.py` - PSI drift utilities.
- `retrox/explain/shap_report.py` - permutation-based importance report.
- `retrox/cli/train.py` - training CLI.
- `retrox/cli/evaluate.py` - evaluation CLI.
- `retrox/cli/drift.py` - drift CLI.
- `retrox/cli/explain.py` - explain CLI.
