from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrox.config import settings
from retrox.data.dengai import load_dengai_train
from retrox.features.engineering import FeatureParams
from retrox.models.training import time_series_cv_scores


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate RetroX model setup with time-series CV.")
    p.add_argument("--city", choices=["sj", "iq"], default=settings.default_city)
    p.add_argument("--horizon", type=int, default=settings.default_horizon_weeks)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="Optional path to write metrics JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_dengai_train(city=args.city, root=args.data_root)
    metrics = time_series_cv_scores(df, horizon_weeks=int(args.horizon), params=FeatureParams())

    print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

