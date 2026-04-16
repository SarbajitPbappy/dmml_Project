from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

City = Literal["sj", "iq"]


class WeeklyRecord(BaseModel):
    year: int | None = None
    weekofyear: int | None = None
    week_start_date: date | None = None

    # climate (any subset; loader will coalesce)
    reanalysis_air_temp_k: float | None = None
    reanalysis_avg_temp_k: float | None = None
    station_avg_temp_c: float | None = None

    reanalysis_specific_humidity_g_per_kg: float | None = None
    reanalysis_relative_humidity_percent: float | None = None

    precipitation_amt_mm: float | None = None
    reanalysis_precip_amt_kg_per_m2: float | None = None

    ndvi_ne: float | None = None
    ndvi_nw: float | None = None
    ndvi_se: float | None = None
    ndvi_sw: float | None = None
    ndvi_mean: float | None = None

    total_cases: float | None = None


class ForecastRequest(BaseModel):
    city: City = "sj"
    horizon_weeks: int = Field(default=4, ge=1, le=12)
    history: list[WeeklyRecord] = Field(
        ..., min_length=8, description="At least 8 weeks recommended for rolling features."
    )


class ForecastResponse(BaseModel):
    city: City
    horizon_weeks: int
    predicted_cases: float
    ers_level: Literal["normal", "elevated", "high"]
    eri_latest: float | None = None


class WhatIfRequest(ForecastRequest):
    override: WeeklyRecord = Field(
        ...,
        description="Values in this record override the latest history record for simulation.",
    )

