"""
training.py — Multi-model training with automatic best-model selection.

Models compared (time-series CV):
  1. Gradient Boosting (HistGradientBoostingRegressor) — fast, handles NaNs natively
  2. Random Forest
  3. Extra Trees Regressor
  4. Ridge Regression (baseline)

The best model (lowest CV MAE) is saved as the artifact.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from retrox.features.engineering import (
    FeatureParams,
    build_supervised_frame,
    clean_numeric_frame,
    numeric_feature_columns,
)
from retrox.logging_utils import get_logger
from retrox.models.registry import ModelArtifact, default_model_dir, save_artifact

logger = get_logger(__name__)


def _to_numeric_df(X: pd.DataFrame, *, feature_cols: list[str]) -> pd.DataFrame:
    return clean_numeric_frame(X, feature_cols)


def _as_frame_and_target(sup) -> tuple[pd.DataFrame, str, list[str] | None]:
    if hasattr(sup, "frame") and hasattr(sup, "target_column"):
        frame = sup.frame
        target = sup.target_column
        feature_cols = list(getattr(sup, "feature_columns", [])) or None
    else:
        frame = sup
        target = "y"
        feature_cols = None
    return frame, target, feature_cols


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Shared pre-processing step: type cast + median impute."""
    to_numeric = FunctionTransformer(
        _to_numeric_df,
        kw_args={"feature_cols": feature_cols},
        validate=False,
    )
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[
                    ("to_numeric", to_numeric),
                    ("impute", SimpleImputer(strategy="median")),
                ]),
                feature_cols,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


logger = get_logger(__name__)


def _candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Return base estimators for candidates (Environment-safe models)."""
    return {
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        "Ridge": Ridge(),
    }


def _tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    *,
    n_trials: int = 5,
    random_state: int = 42,
) -> Any:
    """Use Optuna to find better hyperparams for a model."""
    
    def objective(trial):
        if name == "HistGradientBoosting":
            params = {
                "max_iter": trial.suggest_int("max_iter", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            }
            model = HistGradientBoostingRegressor(**params, random_state=random_state)
        elif name == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = GradientBoostingRegressor(**params, random_state=random_state)
        elif name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            }
            model = RandomForestRegressor(**params, random_state=random_state, n_jobs=-1)
        elif name == "ExtraTrees":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 25),
            }
            model = ExtraTreesRegressor(**params, random_state=random_state, n_jobs=-1)
        else: # Ridge
            params = {"alpha": trial.suggest_float("alpha", 0.1, 200.0, log=True)}
            model = Ridge(**params)

        # TimeSeries CV
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr, te in tscv.split(X):
            pipe = _build_pipeline(model, feature_cols)
            # Check for empty sequences
            if len(tr) < 10 or len(te) < 1:
                continue
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            scores.append(mean_absolute_error(y.iloc[te], pred))
        return np.mean(scores) if scores else 9999.0

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(**best_params, random_state=random_state)
    elif name == "GradientBoosting":
        return GradientBoostingRegressor(**best_params, random_state=random_state)
    elif name == "RandomForest":
        return RandomForestRegressor(**best_params, random_state=random_state, n_jobs=-1)
    elif name == "ExtraTrees":
        return ExtraTreesRegressor(**best_params, random_state=random_state, n_jobs=-1)
    return Ridge(**best_params)


def _build_pipeline(estimator: Any, feature_cols: list[str]) -> Pipeline:
    return Pipeline(steps=[
        ("pre", _build_preprocessor(feature_cols)),
        ("model", estimator),
    ])


def _cv_scores_for_estimator(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    estimator: Any,
    *,
    n_splits: int = 5,
) -> dict[str, float]:
    """Run time-series cross-validation for a single estimator."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    for tr, te in tscv.split(X):
        # We need a fresh instance of the estimator for each fold
        try:
            params = estimator.get_params()
            if "random_state" in params:
                params["random_state"] = 42
            new_est = estimator.__class__(**params)
        except:
            new_est = estimator # Fallback for odd estimators
            
        pipe = _build_pipeline(new_est, feature_cols)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        yt = y.iloc[te].to_numpy()
        maes.append(float(mean_absolute_error(yt, pred)))
        rmses.append(float(np.sqrt(mean_squared_error(yt, pred))))
        r2s.append(float(r2_score(yt, pred)))
    return {
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)),
        "splits": float(n_splits),
    }


def compare_models(
    df: pd.DataFrame,
    *,
    horizon_weeks: int,
    params: FeatureParams | None = None,
    n_splits: int = 5,
    target_column: str | None = None,
) -> dict[str, dict[str, float]]:
    sup = build_supervised_frame(df, horizon_weeks=horizon_weeks, params=params, target_column=target_column)
    sup_frame, target_col, feature_cols = _as_frame_and_target(sup)
    feature_cols = feature_cols or numeric_feature_columns(sup_frame)
    X = sup_frame[feature_cols]
    y = sup_frame[target_col].astype(float)

    candidates = _candidate_models()
    results: dict[str, dict[str, float]] = {}
    for name, estimator in candidates.items():
        try:
            scores = _cv_scores_for_estimator(X, y, feature_cols, estimator, n_splits=n_splits)
            results[name] = scores
            logger.info("CV [%s]: MAE=%.3f RMSE=%.3f R2=%.3f", name, scores["mae"], scores["rmse"], scores["r2"])
        except Exception as exc:
            logger.warning("Model %s failed CV: %s", name, exc)
    return results


def train_model(
    df: pd.DataFrame,
    *,
    city: str,
    horizon_weeks: int,
    params: FeatureParams | None = None,
    random_state: int = 42,
    artifact_name: str = "random_forest",
    extra_meta: dict[str, Any] | None = None,
    target_column: str | None = None,
    n_tuning_trials: int = 3, # Fast tuning
    fast_mode: bool = False,
) -> ModelArtifact:
    """
    Train all candidate models with tuning, select best, and save.
    """
    sup = build_supervised_frame(df, horizon_weeks=horizon_weeks, params=params, target_column=target_column)
    sup_frame, target_col, feature_cols = _as_frame_and_target(sup)
    feature_cols = feature_cols or numeric_feature_columns(sup_frame)
    X = sup_frame[feature_cols]
    y = sup_frame[target_col].astype(float)

    candidate_names = list(_candidate_models().keys())
    
    if fast_mode:
        candidate_names = ["HistGradientBoosting", "RandomForest"]
        n_tuning_trials = 1
        
    # ── Tune & Compare ────────────────────────────────────────────────────────
    all_cv: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_mae = float("inf")
    best_estimator = None

    for name in candidate_names:
        try:
            logger.info("Tuning and evaluating %s ...", name)
            tuned_est = _tune_model(name, X, y, feature_cols, n_trials=n_tuning_trials, random_state=random_state)
            scores = _cv_scores_for_estimator(X, y, feature_cols, tuned_est)
            all_cv[name] = scores
            logger.info("CV [%s]: MAE=%.3f R2=%.3f", name, scores["mae"], scores["r2"])
            if scores["mae"] < best_mae:
                best_mae = scores["mae"]
                best_name = name
                best_estimator = tuned_est
        except Exception as exc:
            logger.warning("Model %s failed: %s", name, exc)

    if best_name is None:
        best_name = "HistGradientBoosting"
        best_estimator = HistGradientBoostingRegressor(random_state=random_state)

    logger.info("Best model selected: %s (MAE=%.3f)", best_name, best_mae)

    # ── Retrain best model on full data ────────────────────────────────────────
    pipe = _build_pipeline(best_estimator, feature_cols)
    pipe.fit(X, y)

    # Feature Importance for SHAP if supported
    # (Optional: we can store a background sample in meta for SHAP dashboard)
    background_sample = X.sample(min(100, len(X))).to_dict(orient="list") if not X.empty else {}

    meta: dict[str, Any] = {
        "city": city,
        "horizon_weeks": horizon_weeks,
        "model_type": best_name,
        "feature_params": asdict(params or FeatureParams(target_column=target_column or "total_cases")),
        "feature_columns": feature_cols,
        "target_column": target_col,
        "cv": all_cv.get(best_name, {}),
        "all_model_cv": all_cv,
        "best_model": best_name,
        "background_sample": background_sample,
    }
    if extra_meta:
        meta.update(extra_meta)

    out_dir = default_model_dir(city=city, horizon_weeks=horizon_weeks)
    artifact = save_artifact(model=pipe, meta=meta, out_dir=out_dir, name=artifact_name)

    logger.info("Saved %s model to %s", best_name, artifact.model_path)
    return artifact

