from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

from retrox.data.dengai import load_dengai_train
from retrox.features.engineering import FeatureParams, build_supervised_frame, clean_numeric_frame
from retrox.models.inference import load_latest_artifact
from retrox.models.registry import default_model_dir


def generate_shap_report(
    *,
    city: str,
    horizon_weeks: int,
    model_name: str = "random_forest",
    out_dir: Path | None = None,
    max_rows: int = 800,
) -> dict[str, str]:
    """
    Generate explainability artifacts for the trained model.

    Writes:
    - feature importance csv
    - top feature markdown summary
    """
    model_dir = default_model_dir(city=city, horizon_weeks=horizon_weeks)
    artifact = load_latest_artifact(model_dir, name=model_name)
    model = artifact.load_model()
    meta = artifact.load_meta()
    feature_cols: list[str] = list(meta["feature_columns"])

    df = load_dengai_train(city=city)
    sup = build_supervised_frame(df, horizon_weeks=horizon_weeks, params=FeatureParams())
    sup_frame = sup.frame if hasattr(sup, "frame") else sup
    X = clean_numeric_frame(sup_frame[feature_cols], feature_cols).iloc[-max_rows:].copy()

    # Use permutation importance on the fitted pipeline for robust compatibility.
    pre = model.named_steps["pre"]
    Xp = pre.transform(X)
    y = sup_frame[meta.get("target_column", "target_total_cases_t_plus_4")].astype(float).iloc[-len(X) :]
    perm = permutation_importance(
        model.named_steps["model"],
        Xp,
        y,
        n_repeats=8,
        random_state=42,
        scoring="neg_mean_absolute_error",
    )

    out_dir = out_dir or (model_dir / "explain" / "shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_abs = perm.importances_mean
    imp = (
        pd.DataFrame({"feature": feature_cols, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = out_dir / "shap_importance.csv"
    imp.to_csv(csv_path, index=False)
    md_path = out_dir / "top_features.md"
    lines = ["# Top Feature Importance", ""]
    for _, row in imp.head(20).iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.6f}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {"summary_png": str(md_path), "importance_csv": str(csv_path)}

