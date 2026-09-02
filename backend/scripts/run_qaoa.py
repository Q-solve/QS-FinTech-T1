"""Run and serialize the local finite-shot Remit-Q QAOA experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.optimization.models import Benchmark, ObjectiveWeights
from backend.app.optimization.qaoa_experiment import (
    REFERENCE_CORRIDOR,
    REFERENCE_PERIOD,
    REFERENCE_SEEDS,
    run_qaoa_experiment,
)
from backend.app.optimization.qaoa_models import QaoaMixer
from backend.app.optimization.qubo import DEFAULT_QUBO_PENALTY


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local QAOA against the validated Remit-Q QUBO."
    )
    parser.add_argument("--period", default=REFERENCE_PERIOD)
    parser.add_argument("--corridor", default=REFERENCE_CORRIDOR)
    parser.add_argument("--benchmark", choices=("CC1", "CC2"), default="CC2")
    parser.add_argument("--fee-weight", type=float, default=0.4)
    parser.add_argument("--fx-weight", type=float, default=0.3)
    parser.add_argument("--speed-weight", type=float, default=0.3)
    parser.add_argument("--penalty", type=float, default=DEFAULT_QUBO_PENALTY)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=2_048)
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument(
        "--mixer",
        choices=tuple(item.value for item in QaoaMixer),
        default=QaoaMixer.STANDARD_X.value,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(REFERENCE_SEEDS))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output = args.output or Path(
        "experiments/results/"
        + (
            f"qaoa_xy_p{args.reps}.json"
            if args.mixer == QaoaMixer.CONSTRAINT_PRESERVING_XY.value
            else f"qaoa_reference_p{args.reps}.json"
        )
    )
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"Refusing to overwrite existing result {output}; choose another --output."
        )
    result = run_qaoa_experiment(
        period=args.period,
        corridor=args.corridor,
        benchmark=Benchmark(args.benchmark),
        weights=ObjectiveWeights(
            args.fee_weight,
            args.fx_weight,
            args.speed_weight,
        ),
        penalty=args.penalty,
        reps=args.reps,
        shots=args.shots,
        max_iterations=args.max_iterations,
        seeds=tuple(args.seeds),
        mixer=QaoaMixer(args.mixer),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    aggregate = result.aggregate
    print(f"Saved {len(result.runs)} run(s) to {output}")
    for run in result.runs:
        state = run.best_feasible_sampled_state
        print(
            f"seed={run.metrics.algorithm_seed} raw="
            f"{run.raw_most_probable_state.qiskit_state_label_xn_to_x0} "
            f"raw_feasible={run.raw_most_probable_state.feasible} "
            f"best_feasible={state.bit_string_x0_to_xn if state else 'none'} "
            f"gap={run.absolute_optimality_gap} "
            f"feasible_probability={run.feasible_probability:.6f} "
            f"optimal_probability={run.optimal_state_probability:.6f}"
        )
    print(
        "exact_optimum_recovery="
        f"{aggregate.exact_optimum_recovery_count}/{aggregate.run_count} "
        f"({aggregate.exact_optimum_recovery_rate:.1%})"
    )
    print(
        "Runtime values are experimental implementation timings, not speedup evidence."
    )


if __name__ == "__main__":
    main()
