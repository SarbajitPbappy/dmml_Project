from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrox.features.engineering import FeatureParams, make_feature_frame


def psi(expected: pd.Series, actual: pd.Series, *, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for drift monitoring.
    Rules of thumb:
    - < 0.10: no drift
    - 0.10–0.25: moderate drift
    - > 0.25: significant drift
    """
    e = pd.to_numeric(expected, errors="coerce").dropna().to_numpy()
    a = pd.to_numeric(actual, errors="coerce").dropna().to_numpy()
    if len(e) < 50 or len(a) < 50:
        return float("nan")

    qs = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(e, qs)
    cuts[0] = -np.inf
    cuts[-1] = np.inf

    def _dist(x: np.ndarray) -> np.ndarray:
        h, _ = np.histogram(x, bins=cuts)
        p = h / max(1, h.sum())
        return np.clip(p, 1e-6, 1.0)

    pe = _dist(e)
    pa = _dist(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


@dataclass(frozen=True)
class DriftResult:
    metric: str
    psi: float


def drift_report(
    reference_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    *,
    params: FeatureParams | None = None,
) -> list[DriftResult]:
    """
    Compute drift scores for key signals (ERI inputs + ERI itself).
    """
    ref_feat = make_feature_frame(reference_df, params=params or FeatureParams())
    rec_feat = make_feature_frame(recent_df, params=params or FeatureParams())

    metrics = ["temp_signal", "humidity_signal", "rain_signal", "ndvi_signal", "eri"]
    out: list[DriftResult] = []
    for m in metrics:
        if m in ref_feat.columns and m in rec_feat.columns:
            out.append(DriftResult(metric=m, psi=psi(ref_feat[m], rec_feat[m])))
    return out

