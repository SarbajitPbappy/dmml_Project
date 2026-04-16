from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException

from retrox.api.schemas import ForecastRequest, ForecastResponse, WhatIfRequest
from retrox.features.engineering import FeatureParams, last_complete_feature_row
from retrox.features.ers import classify_ers
from retrox.models.inference import load_latest_artifact, predict_next
from retrox.models.registry import default_model_dir

app = FastAPI(title="RetroX Dengue Forecast API", version="0.1.0")


def _records_to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")
    return df


def _load_artifact(*, city: str, horizon_weeks: int):
    model_dir = default_model_dir(city=city, horizon_weeks=horizon_weeks)
    return load_latest_artifact(model_dir, name="random_forest")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    try:
        df = _records_to_df([r.model_dump() for r in req.history])
        artifact = _load_artifact(city=req.city, horizon_weeks=req.horizon_weeks)
        pred = predict_next(
            history_df=df,
            artifact=artifact,
            horizon_weeks=req.horizon_weeks,
            params=FeatureParams(),
        )
        feat_last = last_complete_feature_row(df, horizon_weeks=req.horizon_weeks, params=FeatureParams())
        eri_latest = float(feat_last.get("eri", pd.Series([None])).iloc[0]) if "eri" in feat_last.columns else None
        return ForecastResponse(
            city=req.city,
            horizon_weeks=req.horizon_weeks,
            predicted_cases=float(pred),
            ers_level=classify_ers(pred),
            eri_latest=eri_latest,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/whatif", response_model=ForecastResponse)
def whatif(req: WhatIfRequest):
    try:
        base = [r.model_dump() for r in req.history]
        override = req.override.model_dump(exclude_unset=True, exclude_none=True)
        if not base:
            raise ValueError("history is empty")
        base[-1] = {**base[-1], **override}
        df = _records_to_df(base)
        artifact = _load_artifact(city=req.city, horizon_weeks=req.horizon_weeks)
        pred = predict_next(
            history_df=df,
            artifact=artifact,
            horizon_weeks=req.horizon_weeks,
            params=FeatureParams(),
        )
        feat_last = last_complete_feature_row(df, horizon_weeks=req.horizon_weeks, params=FeatureParams())
        eri_latest = float(feat_last.get("eri", pd.Series([None])).iloc[0]) if "eri" in feat_last.columns else None
        return ForecastResponse(
            city=req.city,
            horizon_weeks=req.horizon_weeks,
            predicted_cases=float(pred),
            ers_level=classify_ers(pred),
            eri_latest=eri_latest,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

