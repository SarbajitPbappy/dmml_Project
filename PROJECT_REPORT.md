# RetroX Project Report

## 1) Executive Summary
RetroX is an end-to-end, climate-driven dengue forecasting and early-warning platform designed for public health operations.  
It predicts weekly dengue cases with a 4-week lead time, explains predictions via SHAP, and converts forecasts into actionable alert levels (ERS: Normal, Elevated, High).

This implementation extends academic scope into product scope:
- reproducible ML pipeline
- explainable AI artifacts
- deployable API and dashboard
- data intelligence lab for arbitrary uploaded datasets
- preprocessing export pipeline for ML-ready data
- basic drift monitoring for production retraining triggers

## 2) Problem Statement
Reactive outbreak response increases mortality and strains health systems.  
A practical warning system should use weather and ecological signals to anticipate outbreaks before case spikes are visible in clinics.

## 3) Objectives
- Predict dengue outbreaks 4 weeks ahead.
- Build ERI (composite climate risk score in [0,1]).
- Build ERS (3-level response classification).
- Ensure explainability and trust with SHAP.
- Provide deployable tools for technical and non-technical users.

## 4) Dataset
Primary source: Kaggle DengAI (San Juan + Iquitos weekly records).  
Detected files in this project:
- `data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv`
- `data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv`
- `data/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv`
- `data/DengAI_Predicting_Disease_Spread_-_Submission_Format.csv`

## 5) Methodology
### 5.1 Feature Engineering
- lag features (1–4 weeks) for climate and ERI
- rolling means (4 and 8 weeks)
- autoregressive epidemiology (`cases_lag1`, `cases_lag2`, trend)

### 5.2 ERI (Epidemic Risk Index)
Weighted climate composite:
- Temperature: 0.35
- Humidity: 0.30
- Rainfall: 0.25
- NDVI: 0.10

### 5.3 ERS (Epidemic Risk Score)
- Normal: `< 39`
- Elevated: `39–71`
- High: `> 71`

### 5.4 Modeling & Validation
- Primary model: RandomForestRegressor
- Validation: TimeSeriesSplit (prevents temporal leakage)
- Metrics: MAE, RMSE, R²

### 5.5 Explainability
- SHAP summary plot
- feature-level mean absolute SHAP importance export

## 6) System Architecture
- Data loader: flexible filename/path auto-detection
- Feature pipeline: deterministic engineering + ERI/ERS
- Training CLI: artifact + metadata generation
- API: FastAPI forecast + what-if simulation endpoints
- UI: Streamlit for forecasting + data lab
- Ops: PSI drift report for monitoring

## 7) Advanced Product Features Added
### 7.1 Data Lab (Hackathon-grade extension)
Users can upload arbitrary datasets (`csv/xlsx/json/parquet`) and get:
- schema audit
- missingness/duplicates quality checks
- numeric summary
- correlation heatmap
- distribution + box plots
- automated preprocessing controls:
  - imputation strategy
  - scaling strategy
  - outlier winsorization
  - one-hot encoding
- one-click download of processed dataset (CSV/Parquet when available)

### 7.2 Deployment Readiness
- Dockerfile for API containerization
- Makefile shortcuts
- artifactized model + metadata
- reproducible environment setup

## 8) Expected Operational Workflow
1. Ingest weekly climate + case records.
2. Run model inference for 4-week ahead cases.
3. Convert forecast to ERS level.
4. Trigger intervention playbook:
   - surveillance intensification
   - vector control
   - resource/staff pre-positioning
   - public communication campaign

## 9) Limitations
- Original DengAI geography mismatch vs Bangladesh deployment.
- Social determinants (sanitation, mobility, population density) currently excluded from core model.
- Extreme climate shifts may require periodic retraining.

## 10) Future Work
- add social + mobility + healthcare access covariates
- probabilistic forecasting with uncertainty calibration
- active drift-triggered retraining automation
- district-level Bangladesh localization
- human-in-the-loop intervention feedback loop

## 11) Conclusion
RetroX now functions as a full-stack, explainable, deployment-capable outbreak intelligence platform, not just a notebook experiment.  
It is suitable for final project demonstration, portfolio use, and hackathon deployment pitches.

