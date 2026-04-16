from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib

from retrox.config import settings


def default_model_dir(*, city: str, horizon_weeks: int) -> Path:
    return settings.resolved_artifacts_dir() / "models" / f"city={city}" / f"h={horizon_weeks}"


@dataclass(frozen=True)
class ModelArtifact:
    model_path: Path
    meta_path: Path

    def load_model(self):
        return joblib.load(self.model_path)

    def load_meta(self) -> dict:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))


def save_artifact(*, model, meta: dict, out_dir: Path, name: str) -> ModelArtifact:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{name}.joblib"
    meta_path = out_dir / f"{name}.meta.json"

    meta = dict(meta)
    meta.setdefault("created_at", datetime.now(UTC).isoformat())

    joblib.dump(model, model_path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return ModelArtifact(model_path=model_path, meta_path=meta_path)

