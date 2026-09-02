"""Inspect and exactly validate the provider-selection QUBO."""

from __future__ import annotations

import argparse
from math import isclose

from backend.app.config import CLEANED_DATASET_PATH
from backend.app.optimization.classical_solver import (
    assert_solver_agreement,
    solve_direct_argmin,
    solve_exhaustive_binary,
)
from backend.app.optimization.data_loader import load_cleaned_dataset
from backend.app.optimization.filters import build_eligible_alternatives
from backend.app.optimization.models import Benchmark, ObjectiveWeights
from backend.app.optimization.qubo import (
    DEFAULT_QUBO_PENALTY,
    build_qubo_model,
    is_penalty_sufficient,
    minimum_safe_penalty_threshold,
)
from backend.app.optimization.qubo_solver import (
    solve_qubo_by_enumeration,
    solve_qubo_with_exact_eigensolver,
    to_ising,
    validate_all_qubo_energies,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the exact Remit-Q QUBO.")
    parser.add_argument("--period", default="2025_3Q")
    parser.add_argument("--corridor", default="KENTZA")
    parser.add_argument("--benchmark", choices=("CC1", "CC2"), default="CC2")
    parser.add_argument("--fee-weight", type=float, default=0.4)
    parser.add_argument("--fx-weight", type=float, default=0.3)
    parser.add_argument("--speed-weight", type=float, default=0.3)
    parser.add_argument("--penalty", type=float, default=DEFAULT_QUBO_PENALTY)
    return parser


def main() -> None:
    args = _parser().parse_args()
    weights = ObjectiveWeights(args.fee_weight, args.fx_weight, args.speed_weight)
    alternatives = build_eligible_alternatives(
        load_cleaned_dataset(CLEANED_DATASET_PATH),
        period=args.period,
        corridor=args.corridor,
        benchmark=Benchmark(args.benchmark),
    )
    direct = solve_direct_argmin(alternatives, weights)
    exhaustive = solve_exhaustive_binary(alternatives, weights)
    assert_solver_agreement(direct, exhaustive)

    model = build_qubo_model(alternatives, weights, penalty=args.penalty)
    validation = validate_all_qubo_energies(model)
    enumerated = solve_qubo_by_enumeration(model)
    eigen = solve_qubo_with_exact_eigensolver(model)
    operator, ising_offset = to_ising(model)

    selected_id = direct.selected.alternative.alternative_id
    methods_agree = (
        enumerated.selected_alternative is not None
        and eigen.selected_alternative is not None
        and enumerated.selected_alternative.alternative_id == selected_id
        and eigen.selected_alternative.alternative_id == selected_id
        and isclose(
            enumerated.energy,
            direct.selected.weighted_score,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    )

    print(f"{args.period} {args.corridor} {args.benchmark} QUBO inspection")
    print("variable_mapping (bit strings below are x_0 -> x_n):")
    for variable, alternative, score in zip(
        model.representation.variable_order,
        model.alternatives,
        model.representation.scores,
        strict=True,
    ):
        print(
            f"  {variable}: {alternative.alternative_id} | {alternative.firm} | "
            f"{alternative.payment_instrument} -> {alternative.pickup_method} | "
            f"score={score:.12f}"
        )
    print(f"penalty: {model.representation.penalty:.12f}")
    print(
        "minimum_safe_penalty_condition: P > "
        f"{minimum_safe_penalty_threshold(model.representation.scores):.12f}"
    )
    print(
        "penalty_sufficient: "
        f"{str(is_penalty_sufficient(model.representation.scores, args.penalty)).lower()}"
    )
    print(f"binary_variables_qubits: {len(model.alternatives)}")
    print(f"linear_terms: {len(model.representation.linear_coefficients)}")
    print(f"quadratic_terms: {len(model.representation.quadratic_coefficients)}")
    print(f"constant_offset: {model.representation.constant_offset:.12f}")
    print(f"ising_terms: {len(operator)}")
    print(f"ising_offset: {ising_offset:.12f}")
    print(f"states_energy_validated: {validation.states_checked}")
    print(f"maximum_energy_difference: {validation.maximum_absolute_difference:.3e}")
    print(f"exact_minimum_bit_string_x_order: {enumerated.bit_string}")
    print(f"qiskit_state_label_xn_to_x0: {eigen.qiskit_state_label}")
    selected = enumerated.selected_alternative
    if selected is None:
        print("selected: none")
    else:
        print(
            f"selected: {selected.firm} | {selected.payment_instrument} -> "
            f"{selected.pickup_method} | {selected.alternative_id}"
        )
    print(f"qubo_energy: {enumerated.energy:.12f}")
    print(
        f"original_feasible_weighted_score: {enumerated.original_weighted_score:.12f}"
    )
    print(f"feasible_exactly_one: {str(enumerated.feasible).lower()}")
    print(f"selected_bit_count: {sum(enumerated.bit_values)}")
    print(f"all_four_exact_methods_agree: {str(methods_agree).lower()}")
    if not methods_agree:
        raise SystemExit("Exact methods did not agree.")


if __name__ == "__main__":
    main()
