from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETROX_", extra="ignore")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_dir: Path | None = None
    artifacts_dir: Path | None = None

    eri_w_temperature: float = 0.35
    eri_w_humidity: float = 0.30
    eri_w_rainfall: float = 0.25
    eri_w_ndvi: float = 0.10

    ers_normal_max: int = 38
    ers_elevated_max: int = 71

    default_city: Literal["sj", "iq"] = "sj"
    default_horizon_weeks: int = 4

    def resolved_data_dir(self) -> Path:
        return (self.data_dir or (self.project_root / "data")).resolve()

    def resolved_artifacts_dir(self) -> Path:
        return (self.artifacts_dir or (self.project_root / "artifacts")).resolve()


settings = Settings()

