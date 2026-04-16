from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrox.config import settings


@dataclass(frozen=True)
class ERIWeights:
    temperature: float = settings.eri_w_temperature
    humidity: float = settings.eri_w_humidity
    rainfall: float = settings.eri_w_rainfall
    ndvi: float = settings.eri_w_ndvi


def _minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn = np.nanmin(s.to_numpy())
    mx = np.nanmax(s.to_numpy())
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - mn) / (mx - mn)


def compute_eri(
    df: pd.DataFrame,
    *,
    temperature_col: str,
    humidity_col: str,
    rainfall_col: str,
    ndvi_col: str,
    weights: ERIWeights | None = None,
    out_col: str = "eri",
) -> pd.DataFrame:
    """
    Compute ERI (0..1) from 4 normalized climate signals.

    Uses min-max normalization within the provided dataframe to keep ERI interpretable.
    For deployment, you can compute normalization on training data and reuse, but this
    lightweight approach is robust for demo + coursework.
    """
    w = weights or ERIWeights()
    wsum = w.temperature + w.humidity + w.rainfall + w.ndvi
    if wsum <= 0:
        raise ValueError("ERI weights must sum to a positive value.")

    temp_n = _minmax(df[temperature_col])
    hum_n = _minmax(df[humidity_col])
    rain_n = _minmax(df[rainfall_col])
    ndvi_n = _minmax(df[ndvi_col])

    eri = (
        w.temperature * temp_n
        + w.humidity * hum_n
        + w.rainfall * rain_n
        + w.ndvi * ndvi_n
    ) / wsum
    out = df.copy()
    out[out_col] = eri.clip(0.0, 1.0)
    return out

