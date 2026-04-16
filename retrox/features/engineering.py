from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from retrox.features.eri import compute_eri

ID_COLUMNS = ["city", "year", "weekofyear", "week_start_date"]


@dataclass(frozen=True)
class SupervisedFrame:
    frame: pd.DataFrame
    feature_columns: list[str]
    id_columns: list[str]
    target_column: str


@dataclass(frozen=True)
class FeatureParams:
    lags: tuple[int, ...] = (1, 2, 4, 8, 12, 26, 52)
    rolling_windows: tuple[int, ...] = (4, 8, 12, 26)
    include_autoregressive: bool = True
    target_column: str = "total_cases" # Default for DengAI
    date_column: str | None = "week_start_date"


def infer_signal_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    lower_lookup = {column.lower(): column for column in df.columns}

    def _match(*patterns: str) -> list[str]:
        matched = []
        for lower_name, original_name in lower_lookup.items():
            if any(pattern in lower_name for pattern in patterns):
                matched.append(original_name)
        return matched

    return {
        "temperature": _dedupe(_match("temp", "temperature", "dew_point")),
        "humidity": _dedupe(_match("humidity")),
        "rainfall": _dedupe(_match("precip", "rain")),
        "ndvi": _dedupe(_match("ndvi", "vegetation")),
    }


def _dedupe(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column not in seen:
            ordered.append(column)
            seen.add(column)
    return ordered


def _fill_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = out.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        return out

    # For time-series, interpolation is better than just median
    interpolated = out[numeric_columns].interpolate(limit_direction="both")
    out[numeric_columns] = interpolated.ffill().bfill()
    return out


def _mean_signal(df: pd.DataFrame, columns: list[str], *, fallback: float = 0.0) -> pd.Series:
    if not columns:
        return pd.Series(fallback, index=df.index, dtype=float)
    return df[columns].astype(float).mean(axis=1, skipna=True)


def make_feature_frame(
    df: pd.DataFrame,
    *,
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """
    Engineer a leakage-safe forecasting frame.
    Supports both DengAI specific features and Universal time-series features.
    """
    cfg = params or FeatureParams()
    lags = cfg.lags
    rolling_windows = cfg.rolling_windows
    target_col = cfg.target_column
    date_col = cfg.date_column
    
    out = df.copy()
    
    # 1. Sort by date if possible (Case-insensitive check)
    col_map = {c.lower(): c for c in out.columns}
    d_col = date_col if date_col and date_col in out.columns else None
    
    if not d_col:
        # Try finding a date candidate
        candidates = [c for c in out.columns if any(k in c.lower() for k in ["date", "time", "timestamp"])]
        if candidates:
            d_col = candidates[0]

    if d_col:
        out[d_col] = pd.to_datetime(out[d_col], errors="coerce")
        out = out.sort_values(d_col).reset_index(drop=True)
    elif "year" in col_map and "weekofyear" in col_map:
        y_c = col_map["year"]
        w_c = col_map["weekofyear"]
        out = out.sort_values([y_c, w_c]).reset_index(drop=True)
    elif "year" in col_map:
        out = out.sort_values([col_map["year"]]).reset_index(drop=True)

    out = _fill_numeric(out)
    
    # 2. Dengue Specific Features (if signals detected)
    groups = infer_signal_columns(out)
    has_dengue_signals = any(groups.values())
    
    if has_dengue_signals:
        out["signal_temperature"] = _mean_signal(out, groups["temperature"])
        out["signal_humidity"] = _mean_signal(out, groups["humidity"])
        out["signal_rainfall"] = _mean_signal(out, groups["rainfall"])
        out["signal_ndvi"] = _mean_signal(out, groups["ndvi"])

        out = compute_eri(
            out,
            temperature_col="signal_temperature",
            humidity_col="signal_humidity",
            rainfall_col="signal_rainfall",
            ndvi_col="signal_ndvi",
        )
    
    # 3. Time-based features
    if "weekofyear" in out.columns:
        week = pd.to_numeric(out["weekofyear"], errors="coerce").fillna(0.0)
        phase = 2.0 * np.pi * week / 52.0
        out["week_sin"] = np.sin(phase)
        out["week_cos"] = np.cos(phase)
    elif date_col and date_col in out.columns:
        week = out[date_col].dt.isocalendar().week.astype(float)
        phase = 2.0 * np.pi * week / 52.0
        out["week_sin"] = np.sin(phase)
        out["week_cos"] = np.cos(phase)

    # 4. Universal Lag & Rolling Features
    # We apply this to all numeric columns that aren't IDs or the target (which has its own logic)
    ignore = set(ID_COLUMNS) | {target_col, "y", "eri", "week_sin", "week_cos"}
    base_features = [c for c in out.columns if c not in ignore and pd.api.types.is_numeric_dtype(out[c])]
    
    # Include ERI if it was computed
    if "eri" in out.columns:
        base_features.append("eri")

    for column in base_features:
        # Lags
        for lag in lags:
            out[f"{column}_lag_{lag}"] = out[column].shift(lag)
        
        # Rolling stats
        for window in rolling_windows:
            if len(out) >= window:
                out[f"{column}_roll_mean_{window}"] = out[column].rolling(window=window, min_periods=window//2).mean()
                out[f"{column}_roll_std_{window}"] = out[column].rolling(window=window, min_periods=window//2).std()
                # Exponential moving average
                out[f"{column}_ewm_mean_{window}"] = out[column].ewm(span=window).mean()

    # 5. Target Autoregressive Features
    if target_col in out.columns:
        # Check if target is truly numeric before applying math operations
        is_num = pd.api.types.is_numeric_dtype(out[target_col]) and not pd.api.types.is_datetime64_any_dtype(out[target_col])
        if is_num:
            # For targets, we use broader lags to capture long-term trends
            for l in lags:
                out[f"{target_col}_lag_{l}"] = out[target_col].shift(l)
            
            for w in rolling_windows:
                out[f"{target_col}_roll_mean_{w}"] = out[target_col].rolling(window=w, min_periods=w//2).mean()
                # Momentum / Trend
                out[f"{target_col}_diff_{w}"] = out[target_col].diff(w)
                
            # Specific Dengue Features if appropriate
            if target_col == "total_cases" and "eri" in out.columns:
                out["outbreak_momentum"] = out["total_cases_roll_mean_4"] * (1.0 + out["eri"])

    return out


def build_supervised_frame(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
    horizon_weeks: int | None = None,
    params: FeatureParams | None = None,
    target_column: str | None = None,
) -> SupervisedFrame:
    if horizon_weeks is not None:
        horizon = int(horizon_weeks)
        
    cfg = params or FeatureParams()
    target_col = target_column or cfg.target_column
    
    # Add target_column to params to ensure make_feature_frame uses it
    new_params = FeatureParams(
        lags=cfg.lags,
        rolling_windows=cfg.rolling_windows,
        include_autoregressive=cfg.include_autoregressive,
        target_column=target_col,
        date_column=cfg.date_column
    )
    
    features = make_feature_frame(df, params=new_params)
    future_target_column = f"target_{target_col}_t_plus_{horizon}"
    
    # The actual prediction target is shifted by horizon
    features[future_target_column] = features[target_col].shift(-horizon)

    # Feature columns: everything numeric except IDs and targets
    feature_columns = [
        column
        for column in features.columns
        if column not in ID_COLUMNS + [target_col, future_target_column]
        and pd.api.types.is_numeric_dtype(features[column])
    ]

    # Quality check: discard rows with too many NaNs in features
    # (Happens at start of sequence due to deep lags)
    feature_quality = features[feature_columns].notna().mean(axis=1)
    valid_rows = features[future_target_column].notna() & (feature_quality >= 0.7)
    frame = features.loc[valid_rows].reset_index(drop=True)
    
    return SupervisedFrame(
        frame=frame,
        feature_columns=feature_columns,
        id_columns=ID_COLUMNS,
        target_column=future_target_column,
    )


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    ignore = set(ID_COLUMNS) | {"total_cases", "y"}
    return [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]


def clean_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def last_complete_feature_row(
    df: pd.DataFrame, *, horizon_weeks: int, params: FeatureParams | None = None
) -> pd.DataFrame:
    _ = horizon_weeks  # kept for API compatibility
    feat = make_feature_frame(df, params=params)
    if feat.empty:
        raise ValueError("No rows available for feature generation.")
    return feat.tail(1).reset_index(drop=True)

