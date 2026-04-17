from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from retrox.config import settings

City = Literal["sj", "iq"]


@dataclass(frozen=True)
class DengAIPaths:
    root: Path

    @property
    def features_train(self) -> Path:
        names = [
            "dengue_features_train.csv",
            "DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv",
        ]
        for n in names:
            p = self.root / n
            if p.exists():
                return p
        return self.root / names[0]

    @property
    def labels_train(self) -> Path:
        names = [
            "dengue_labels_train.csv",
            "DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv",
        ]
        for n in names:
            p = self.root / n
            if p.exists():
                return p
        return self.root / names[0]

    @property
    def features_test(self) -> Path:
        names = [
            "dengue_features_test.csv",
            "DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv",
        ]
        for n in names:
            p = self.root / n
            if p.exists():
                return p
        return self.root / names[0]


def _default_dengai_root() -> Path:
    preferred = settings.resolved_data_dir() / "raw" / "dengai"
    if preferred.exists():
        return preferred
    return settings.resolved_data_dir()


def dengai_files_available(root: Path | None = None) -> bool:
    root = (root or _default_dengai_root()).resolve()
    paths = DengAIPaths(root=root)
    return paths.features_train.exists() and paths.labels_train.exists()


def load_dengai_train(*, city: City, root: Path | None = None) -> pd.DataFrame:
    """
    Load DengAI training data for a city.

    Returns a dataframe sorted by time with:
    - id columns: city, year, weekofyear, week_start_date
    - feature columns: all climate inputs
    - target column: total_cases
    """
    root = (root or _default_dengai_root()).resolve()
    paths = DengAIPaths(root=root)

    if not paths.features_train.exists() or not paths.labels_train.exists():
        raise FileNotFoundError(
            "Missing DengAI files. Expected:\n"
            f"- {paths.features_train}\n"
            f"- {paths.labels_train}\n"
            "Download from Kaggle and place them under data/raw/dengai/."
        )

    features = pd.read_csv(paths.features_train)
    labels = pd.read_csv(paths.labels_train)

    df = features.merge(labels, on=["city", "year", "weekofyear"], how="inner")
    df = df[df["city"] == city].copy()

    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")

    df = df.sort_values(["year", "weekofyear"]).reset_index(drop=True)
    return df


def load_dengai_test(*, city: City, root: Path | None = None) -> pd.DataFrame:
    root = (root or _default_dengai_root()).resolve()
    paths = DengAIPaths(root=root)

    if not paths.features_test.exists():
        raise FileNotFoundError(
            "Missing DengAI test file. Expected:\n"
            f"- {paths.features_test}\n"
            "Download from Kaggle and place it under data/raw/dengai/."
        )

    df = pd.read_csv(paths.features_test)
    df = df[df["city"] == city].copy()
    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    df = df.sort_values(["year", "weekofyear"]).reset_index(drop=True)
    return df


def write_demo_dataset(
    *,
    root: Path | None = None,
    years: int = 18,
    holdout_weeks: int = 16,
    seed: int = 42,
) -> DengAIPaths:
    """
    Create a DengAI-shaped synthetic dataset so the full platform runs without Kaggle.

    The synthetic generator deliberately bakes in:
    - climate seasonality
    - a 2-4 week biological lag
    - autoregressive outbreak momentum
    """
    root = (root or _default_dengai_root()).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = DengAIPaths(root=root)
    rng = np.random.default_rng(seed)

    city_profiles = {
        "sj": {
            "phase": 0.15,
            "temp_c": 28.0,
            "humidity": 77.0,
            "rainfall": 58.0,
            "ndvi": 0.58,
            "trend": 0.9,
        },
        "iq": {
            "phase": 0.65,
            "temp_c": 27.2,
            "humidity": 82.0,
            "rainfall": 88.0,
            "ndvi": 0.73,
            "trend": 1.2,
        },
    }

    def _seasonal_wave(week_idx: int, *, phase: float) -> float:
        return np.sin((2.0 * np.pi * week_idx / 52.0) + phase)

    rows: list[dict[str, float | int | str]] = []
    total_weeks = years * 52
    start_date = pd.Timestamp("2000-01-02")

    for city, profile in city_profiles.items():
        history_temp: list[float] = []
        history_humidity: list[float] = []
        history_rain: list[float] = []
        history_ndvi: list[float] = []
        prev_cases_1 = 18.0
        prev_cases_2 = 14.0

        for week_idx in range(total_weeks):
            date = start_date + pd.Timedelta(weeks=week_idx)
            seasonal = _seasonal_wave(week_idx, phase=profile["phase"])
            shoulder = _seasonal_wave(week_idx, phase=profile["phase"] + 0.9)

            station_avg_temp_c = (
                profile["temp_c"] + 2.1 * seasonal + 0.9 * shoulder + rng.normal(0.0, 0.55)
            )
            station_min_temp_c = station_avg_temp_c - 3.8 + rng.normal(0.0, 0.4)
            station_max_temp_c = station_avg_temp_c + 4.4 + rng.normal(0.0, 0.4)
            station_diur_temp_rng_c = station_max_temp_c - station_min_temp_c

            reanalysis_relative_humidity_percent = (
                profile["humidity"] + 5.5 * shoulder + rng.normal(0.0, 1.8)
            )
            reanalysis_specific_humidity_g_per_kg = (
                reanalysis_relative_humidity_percent / 6.0 + rng.normal(0.0, 0.25)
            )
            reanalysis_dew_point_temp_k = station_avg_temp_c + 273.15 - 1.9 + rng.normal(0.0, 0.25)

            precipitation_amt_mm = max(
                0.0,
                profile["rainfall"] * (1.0 + 0.45 * seasonal + 0.22 * shoulder)
                + rng.normal(0.0, 8.0),
            )
            station_precip_mm = max(0.0, precipitation_amt_mm * 0.72 + rng.normal(0.0, 4.0))
            reanalysis_precip_amt_kg_per_m2 = max(
                0.0, precipitation_amt_mm * 0.86 + rng.normal(0.0, 3.0)
            )
            reanalysis_sat_precip_amt_mm = max(
                0.0, precipitation_amt_mm * 1.05 + rng.normal(0.0, 5.0)
            )

            ndvi_core = np.clip(
                profile["ndvi"] + 0.08 * shoulder + 0.03 * seasonal + rng.normal(0.0, 0.025),
                0.18,
                0.95,
            )
            ndvi_ne = np.clip(ndvi_core + rng.normal(0.0, 0.018), 0.1, 0.98)
            ndvi_nw = np.clip(ndvi_core + rng.normal(0.0, 0.018), 0.1, 0.98)
            ndvi_se = np.clip(ndvi_core + rng.normal(0.0, 0.018), 0.1, 0.98)
            ndvi_sw = np.clip(ndvi_core + rng.normal(0.0, 0.018), 0.1, 0.98)

            lag_temp = history_temp[-3] if len(history_temp) >= 3 else station_avg_temp_c
            lag_humidity = (
                history_humidity[-3]
                if len(history_humidity) >= 3
                else reanalysis_relative_humidity_percent
            )
            lag_rain = history_rain[-2] if len(history_rain) >= 2 else precipitation_amt_mm
            lag_ndvi = history_ndvi[-3] if len(history_ndvi) >= 3 else ndvi_core

            vector_pressure = (
                2.8 * max(lag_temp - 26.0, 0.0)
                + 0.08 * lag_rain
                + 0.33 * max(lag_humidity - 72.0, 0.0)
                + 22.0 * lag_ndvi
            )
            endemic_pressure = 0.46 * prev_cases_1 + 0.19 * prev_cases_2
            seasonal_pressure = 10.0 * (seasonal + 1.1) + profile["trend"] * (week_idx / total_weeks)
            raw_cases = vector_pressure + endemic_pressure + seasonal_pressure + rng.normal(0.0, 6.0) - 28.0
            total_cases = int(np.clip(np.round(raw_cases), 0, 260))

            history_temp.append(station_avg_temp_c)
            history_humidity.append(reanalysis_relative_humidity_percent)
            history_rain.append(precipitation_amt_mm)
            history_ndvi.append(ndvi_core)
            prev_cases_2 = prev_cases_1
            prev_cases_1 = float(total_cases)

            rows.append(
                {
                    "city": city,
                    "year": int(date.isocalendar().year),
                    "weekofyear": int(date.isocalendar().week),
                    "week_start_date": date.strftime("%Y-%m-%d"),
                    "ndvi_ne": ndvi_ne,
                    "ndvi_nw": ndvi_nw,
                    "ndvi_se": ndvi_se,
                    "ndvi_sw": ndvi_sw,
                    "precipitation_amt_mm": precipitation_amt_mm,
                    "reanalysis_air_temp_k": station_avg_temp_c + 273.15 + rng.normal(0.0, 0.25),
                    "reanalysis_avg_temp_k": station_avg_temp_c + 273.15 + rng.normal(0.0, 0.20),
                    "reanalysis_dew_point_temp_k": reanalysis_dew_point_temp_k,
                    "reanalysis_max_air_temp_k": station_max_temp_c + 273.15 + rng.normal(0.0, 0.25),
                    "reanalysis_min_air_temp_k": station_min_temp_c + 273.15 + rng.normal(0.0, 0.25),
                    "reanalysis_precip_amt_kg_per_m2": reanalysis_precip_amt_kg_per_m2,
                    "reanalysis_relative_humidity_percent": reanalysis_relative_humidity_percent,
                    "reanalysis_sat_precip_amt_mm": reanalysis_sat_precip_amt_mm,
                    "reanalysis_specific_humidity_g_per_kg": reanalysis_specific_humidity_g_per_kg,
                    "station_avg_temp_c": station_avg_temp_c,
                    "station_diur_temp_rng_c": station_diur_temp_rng_c,
                    "station_max_temp_c": station_max_temp_c,
                    "station_min_temp_c": station_min_temp_c,
                    "station_precip_mm": station_precip_mm,
                    "total_cases": total_cases,
                }
            )

    full_df = pd.DataFrame(rows).sort_values(["city", "year", "weekofyear"]).reset_index(drop=True)
    train_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []

    for city in city_profiles:
        city_df = full_df[full_df["city"] == city].reset_index(drop=True)
        split_at = max(52, len(city_df) - holdout_weeks)
        train_rows.append(city_df.iloc[:split_at].copy())
        test_rows.append(city_df.iloc[split_at:].drop(columns=["total_cases"]).copy())

    train_df = pd.concat(train_rows, ignore_index=True)
    labels_df = train_df[["city", "year", "weekofyear", "total_cases"]].copy()
    features_train_df = train_df.drop(columns=["total_cases"]).copy()
    features_test_df = pd.concat(test_rows, ignore_index=True)

    features_train_df.to_csv(paths.features_train, index=False)
    labels_df.to_csv(paths.labels_train, index=False)
    features_test_df.to_csv(paths.features_test, index=False)
    return paths
