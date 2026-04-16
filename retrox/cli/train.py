from __future__ import annotations

import argparse
from pathlib import Path

from retrox.config import settings
from retrox.data.dengai import load_dengai_train
from retrox.features.engineering import FeatureParams
from retrox.models.training import train_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train RetroX dengue forecasting model.")
    p.add_argument("--city", choices=["sj", "iq"], default=settings.default_city)
    p.add_argument("--horizon", type=int, default=settings.default_horizon_weeks)
    p.add_argument("--data-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_dengai_train(city=args.city, root=args.data_root)
    train_model(
        df,
        city=args.city,
        horizon_weeks=int(args.horizon),
        params=FeatureParams(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

