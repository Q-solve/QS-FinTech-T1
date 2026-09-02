"""Print a classical ranking for the validated Kenya-to-Tanzania instance."""

from __future__ import annotations

import argparse

from backend.app.config import CLEANED_DATASET_PATH
from backend.app.optimization.classical_solver import (
    assert_solver_agreement,
    solve_direct_argmin,
    solve_exhaustive_binary,
)
from backend.app.optimization.data_loader import load_cleaned_dataset
from backend.app.optimization.filters import build_eligible_alternatives
from backend.app.optimization.models import Benchmark, ObjectiveWeights, SolverResult


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank 2025_3Q KENTZA provider/service alternatives."
    )
    parser.add_argument("--benchmark", choices=("CC1", "CC2"), default="CC2")
    parser.add_argument("--fee-weight", type=float, default=0.4)
    parser.add_argument("--fx-weight", type=float, default=0.3)
    parser.add_argument("--speed-weight", type=float, default=0.3)
    return parser


def _print_result(result: SolverResult) -> None:
    selected = result.selected
    item = selected.alternative
    print(f"solver: {result.solver_name}")
    print(f"runtime_seconds: {result.runtime_seconds:.9f}")
    print(f"eligible_alternatives: {result.eligible_alternatives}")
    print(
        "selected: "
        f"{item.firm} | {item.payment_instrument} | {item.speed_label} | "
        f"{item.pickup_method} [{item.alternative_id}]"
    )
    print(
        f"benchmark: {item.benchmark.value} = "
        f"{item.benchmark_currency} {item.benchmark_amount:,.0f}"
    )
    print(
        "raw: "
        f"fee_percentage={item.fee_percentage:.6f}, "
        f"fx_margin={item.fx_margin:.6f}, "
        f"speed_label={item.speed_label!r}, speed_ordinal={item.speed_ordinal}, "
        f"total_cost_percentage={item.total_cost_percentage:.6f}"
    )
    print(
        "normalized: "
        f"fee={selected.normalized.fee:.6f}, "
        f"fx={selected.normalized.fx:.6f}, "
        f"speed={selected.normalized.speed:.6f}"
    )
    print(
        "weights: "
        f"fee={result.weights.fee_weight:.6f}, "
        f"fx={result.weights.fx_weight:.6f}, "
        f"speed={result.weights.speed_weight:.6f}"
    )
    print(f"weighted_score: {selected.weighted_score:.12f}")
    print(
        f"winning_score_ties: {result.tie_information.tied_count}; "
        f"tie_breaking_applied={result.tie_information.tie_breaking_applied}"
    )


def main() -> None:
    args = _parser().parse_args()
    weights = ObjectiveWeights(
        fee_weight=args.fee_weight,
        fx_weight=args.fx_weight,
        speed_weight=args.speed_weight,
    )
    benchmark = Benchmark(args.benchmark)
    dataset = load_cleaned_dataset(CLEANED_DATASET_PATH)
    alternatives = build_eligible_alternatives(
        dataset, period="2025_3Q", corridor="KENTZA", benchmark=benchmark
    )
    direct = solve_direct_argmin(alternatives, weights)
    exhaustive = solve_exhaustive_binary(alternatives, weights)
    assert_solver_agreement(direct, exhaustive)

    print("2025_3Q KENTZA classical provider/service ranking")
    _print_result(direct)
    print()
    _print_result(exhaustive)
    print("\nranked_alternatives:")
    for ranked in direct.ranked_alternatives:
        item = ranked.alternative
        print(
            f"{ranked.rank:>2}. {item.firm} | {item.payment_instrument} | "
            f"{item.pickup_method} | score={ranked.weighted_score:.12f} | "
            f"fee={item.fee_percentage:.6f}% | fx={item.fx_margin:.6f}% | "
            f"speed={item.speed_label} ({item.speed_ordinal}) | "
            f"total={item.total_cost_percentage:.6f}% | {item.alternative_id}"
        )
    print("\nsolver_agreement: true")


if __name__ == "__main__":
    main()
