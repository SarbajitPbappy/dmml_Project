# 🧠 RetroX Architectural & Technical Documentation

This document provides a deep, structural analysis of the RetroX platform. It breaks down the data acquisition flows, machine learning execution constraints, user interface methodologies, and the analytical features implemented across the system.

---

## 1. Data Engineering & Subsystems (`retrox/features`)

The primary objective of the Feature Pipeline is transforming semi-structured chronological limits into highly regressive matrix frames capable of mapping extensive environmental or arbitrary historical momentum.

### 1.1 The `FeatureParams` Dataclass
All architectural processing originates via the `FeatureParams` specification which defines scaling boundaries:
* **Deep Lags**: Dictates the backward horizons `(1, 2, 4, 8, 12, 26, 52)` weeks translated into shift distributions. Captures both micro-momentum (4 weeks) and deep macro-seasonality (52-week annual recurrent cycles).
* **Rolling Statistics**: Computes expanding geometric mean windows `(4, 8, 12, 26)`.
* **Type-Constraints**: Uses Pandas `is_numeric_dtype` validation to strictly isolate target operations. Non-numeric structures (like Dates, explicit IDs, categorical text) are universally ignored from rolling calculation matrices to prevent mathematical crash triggers.

### 1.2 Imputation & Interpolation Logic
If the dataset lacks structural integrity (missing fields), RetroX manages inference sequentially across `engineering.py` and `lab_pipeline.py`. Initial forward-filling mitigates micro-gaps, while Scikit-Learn's `IterativeImputer` (or fallback `SimpleImputer`) runs structural estimations for heavy NaN gaps without invalidating the target correlations.

---

## 2. Machine Learning Operations (`retrox/models`)

The RetroX Training stack is built completely autonomously to ensure that no programmatic interventions are required from the analyst dashboard. 

### 2.1 The Optuna AutoML Hyper-parameter Hub
Once data is isolated into `SupervisedFrame` matrices, it enters the objective optimization space. The modeling engine uses **Optuna** for Bayesian optimization. It iterates over parameter spaces (like `max_depth`, `learning_rate`, `min_samples_leaf`) minimizing the Validation Mean Absolute Error (MAE).

### 2.2 Dual-Tier Processing Engine
Because tuning arbitrary structural data is intensely bottlenecked, we split the training heuristic in `retrox.models.training.train_model()`:
* **Exhaustive Deep Search**: Standard mode for competition and long-horizon medical analytics. Uses `HistGradientBoosting`, `GradientBoosting`, `RandomForest`, `ExtraTrees` and `Ridge`. Iterates across 10 extensive trials, measuring against a robust `TimeSeriesSplit(n_splits=5)`.
* **Fast Universal Mode (`fast_mode=True`)**: Deployed autonomously when the user targets a universal CSV upload. Downshifts processing exclusively to `HistGradientBoosting` and `RandomForest` constrained by `n_tuning_trials=1`. This reduces execution time by over 95%, ensuring UI fluidity while maintaining strong tree-based predictive metrics.

### 2.3 Checkpointing & Inference
Upon convergence, RetroX isolates the optimal pipeline, exporting serialization through `joblib`. Artifact states implicitly cache parameters including:
* Target Objective.
* Training Dimension.
* Model `Feature Override` (verifying overlap if the user manipulates horizon slider inputs). 
During `predict_next`, the inference function guarantees dimension parity dynamically before unlocking the prediction.

---

## 3. Explanatory & Analytical Intercepts

RetroX transitions away from "Black Box AI" by enforcing strict analytical readouts mapped through the SHAP pipeline.

### 3.1 SHAP Waterfall Verification (`explain_prediction`)
Integrated directly inside `retrox/models/inference.py`, this function generates a localized TreeExplainer interpreting the immediate predictive step ahead. The vector breakdown defines:
* Base Expected Value.
* Exact algorithmic drift derived exclusively from each feature shift (e.g. `signal_temperature_lag_4` pushing total cases +5.20).

### 3.2 Dynamic UI Rendering (`retrox/dashboard/app.py`)
To enable true Universal Data Handling, the frontend engine reacts intelligently to data schemas:
* **Medical Render State**: Operates predominantly when the target parameter is `total_cases`, extracting the complex `ERI (Environmental Risk Index)`.
* **Universal Operations Render State**: Converts telemetry into global actionable indexes like `Anomaly Z-Score` and calculates `Trend Velocity`. 

### 3.3 The Custom Auto-EDA Data Lab (`lab_pipeline.py`)
Before passing into modeling, the dataset traverses a fully automated structural review sequence displaying:
- Complete variance analysis.
- PCA-based (or t-SNE fallback) cluster analysis.
- Automated distribution histograms utilizing native robust Plotly graphing architectures.

---

## Conclusion
The architecture maintains fully containerizable properties out-of-the-box. Due to its dynamic routing handling (`fast_mode`, `is_numeric` safeguards, `subset` verification artifacts), the system can natively deploy into Railway or AWS contexts utilizing nothing but the Python framework and structural entry points defined in the configuration files.
