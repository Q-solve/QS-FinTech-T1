"""Create a machine-readable comparison of stored standard and XY experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.optimization.qaoa_experiment import build_mixer_comparison


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare stored QAOA mixer experiments."
    )
    parser.add_argument(
        "--standard-p1",
        type=Path,
        default=Path("experiments/results/qaoa_reference_p1.json"),
    )
    parser.add_argument(
        "--standard-p2",
        type=Path,
        default=Path("experiments/results/qaoa_reference_p2.json"),
    )
    parser.add_argument(
        "--xy-p1", type=Path, default=Path("experiments/results/qaoa_xy_p1.json")
    )
    parser.add_argument(
        "--xy-p2", type=Path, default=Path("experiments/results/qaoa_xy_p2.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/qaoa_mixer_comparison.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing result {args.output}; choose another --output."
        )
    result = build_mixer_comparison(
        {
            "standard_x_p1": args.standard_p1,
            "constraint_preserving_xy_p1": args.xy_p1,
            "standard_x_p2": args.standard_p2,
            "constraint_preserving_xy_p2": args.xy_p2,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved mixer comparison to {args.output}")


if __name__ == "__main__":
    main()
