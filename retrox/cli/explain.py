from __future__ import annotations

import argparse
import json

from retrox.config import settings
from retrox.explain.shap_report import generate_shap_report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate SHAP explainability report.")
    p.add_argument("--city", choices=["sj", "iq"], default=settings.default_city)
    p.add_argument("--horizon", type=int, default=settings.default_horizon_weeks)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = generate_shap_report(city=args.city, horizon_weeks=int(args.horizon))
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

