from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrox.config import settings
from retrox.data.dengai import load_dengai_train
from retrox.ops.drift import drift_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute PSI drift report (baseline vs recent).")
    p.add_argument("--city", choices=["sj", "iq"], default=settings.default_city)
    p.add_argument("--recent-weeks", type=int, default=52)
    p.add_argument("--data-root", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df = load_dengai_train(city=args.city, root=args.data_root)
    recent = df.tail(int(args.recent_weeks)).reset_index(drop=True)
    ref = df.iloc[: max(0, len(df) - len(recent))].reset_index(drop=True)
    rep = drift_report(ref, recent)
    out = [{"metric": r.metric, "psi": r.psi} for r in rep]
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

