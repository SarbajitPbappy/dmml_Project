import shap
import numpy as np
import pandas as pd
from pathlib import Path

from retrox.features.engineering import (
    FeatureParams,
    last_complete_feature_row,
    numeric_feature_columns,
)
from retrox.models.registry import ModelArtifact


def load_latest_artifact(model_dir: Path, *, name: str = "random_forest") -> ModelArtifact:
    model_path = model_dir / f"{name}.joblib"
    meta_path = model_dir / f"{name}.meta.json"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path} / {meta_path}")
    return ModelArtifact(model_path=model_path, meta_path=meta_path)


def predict_next(
    *,
    history_df: pd.DataFrame,
    artifact: ModelArtifact,
    horizon_weeks: int,
    params: FeatureParams | None = None,
) -> float:
    """Predict the value at horizon_weeks from the last available complete data point."""
    row = last_complete_feature_row(history_df, horizon_weeks=horizon_weeks, params=params)
    meta = artifact.load_meta()
    
    feature_cols = meta.get("feature_columns")
    if not feature_cols:
        feature_cols = numeric_feature_columns(row)

    # ── Robustness Check ──────────────────────────────────────────────────────
    missing = [c for c in feature_cols if c not in row.columns]
    if missing:
        target = meta.get("target_column", "Target")
        msg = (
            f"The loaded model (Target: {target}) requires features {missing[:3]}... "
            "which are missing from your uploaded dataset. "
            "Please train a new artifact for this specific dataset."
        )
        raise ValueError(msg)
    # ──────────────────────────────────────────────────────────────────────────

    X = row[feature_cols]
    model = artifact.load_model()
    pred = model.predict(X)[0]
    return float(max(0, pred))


def explain_prediction(
    *,
    history_df: pd.DataFrame,
    artifact: ModelArtifact,
    horizon_weeks: int,
    params: FeatureParams | None = None,
) -> dict:
    """Compute SHAP values for the most recent prediction."""
    model = artifact.load_model()
    meta = artifact.load_meta()
    feature_cols = meta.get("feature_columns")
    
    # Get the feature row
    row = last_complete_feature_row(history_df, horizon_weeks=horizon_weeks, params=params)
    X = row[feature_cols]
    
    # ── SHAP ─────────────────────────────────────────────────────────────────
    # We use a background sample from meta if available, else a small slice of historical data
    bg_dict = meta.get("background_sample", {})
    if bg_dict:
        bg_data = pd.DataFrame(bg_dict)
    else:
        # Fallback background
        bg_data = X # Not ideal but works
        
    try:
        # We use the final estimator step from the pipeline
        estimator = model.named_steps["model"]
        preprocessor = model.named_steps["pre"]
        
        # Transform data first
        X_tf = preprocessor.transform(X)
        bg_tf = preprocessor.transform(bg_data)
        
        # Most modern trees or linear models
        explainer = shap.Explainer(estimator, bg_tf)
        shap_values = explainer(X_tf)
        
        return {
            "shap_values": shap_values,
            "feature_names": feature_cols,
            "base_value": float(shap_values.base_values[0]) if hasattr(shap_values, "base_values") else 0.0,
        }
    except Exception as e:
        return {"error": str(e)}

