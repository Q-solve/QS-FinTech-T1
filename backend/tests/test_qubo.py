"""Correctness tests for the provider-selection QUBO formulation."""

from __future__ import annotations

from itertools import product
from math import isclose

import pytest
from qiskit_optimization.problems.constraint import Constraint

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
    EXACTLY_ONE_CONSTRAINT,
    build_qubo_model,
    is_penalty_sufficient,
    minimum_safe_penalty_threshold,
    qubo_energy,
)
from backend.app.optimization.qubo_solver import (
    solve_qubo_by_enumeration,
    solve_qubo_with_exact_eigensolver,
    to_ising,
    validate_all_qubo_energies,
)

REFERENCE_WEIGHTS = ObjectiveWeights(0.4, 0.3, 0.3)


@pytest.fixture(scope="module")
def dataset():
    return load_cleaned_dataset(CLEANED_DATASET_PATH)


@pytest.fixture(scope="module")
def cc1_alternatives(dataset):
    return build_eligible_alternatives(
        dataset, period="2025_3Q", corridor="KENTZA", benchmark=Benchmark.CC1
    )


@pytest.fixture(scope="module")
def cc2_alternatives(dataset):
    return build_eligible_alternatives(
        dataset, period="2025_3Q", corridor="KENTZA", benchmark=Benchmark.CC2
    )


@pytest.fixture(scope="module")
def reference_model(cc2_alternatives):
    return build_qubo_model(
        cc2_alternatives,
        REFERENCE_WEIGHTS,
        penalty=DEFAULT_QUBO_PENALTY,
    )


def test_stable_fourteen_variable_mapping(reference_model) -> None:
    expected_ids = tuple(
        sorted(
            alternative.alternative_id for alternative in reference_model.alternatives
        )
    )
    assert len(reference_model.alternatives) == 14
    assert reference_model.representation.variable_order == tuple(
        f"x_{index}" for index in range(14)
    )
    assert reference_model.representation.alternative_ids == expected_ids
    assert (
        tuple(
            alternative.alternative_id for alternative in reference_model.alternatives
        )
        == expected_ids
    )


def test_constrained_program_has_exactly_one_constraint(reference_model) -> None:
    problem = reference_model.constrained_problem
    assert problem.get_num_binary_vars() == 14
    assert problem.get_num_vars() == 14
    assert problem.get_num_linear_constraints() == 1
    constraint = problem.get_linear_constraint(EXACTLY_ONE_CONSTRAINT)
    assert constraint.sense is Constraint.Sense.EQ
    assert constraint.rhs == 1.0
    assert constraint.linear.to_dict(use_name=True) == {
        variable: 1.0 for variable in reference_model.representation.variable_order
    }
    assert problem.objective.linear.to_array() == pytest.approx(
        reference_model.representation.scores
    )


def test_expanded_qubo_coefficients(reference_model) -> None:
    representation = reference_model.representation
    penalty = representation.penalty
    assert representation.constant_offset == penalty
    for variable, score in zip(
        representation.variable_order, representation.scores, strict=True
    ):
        assert representation.linear_coefficients[variable] == pytest.approx(
            score - penalty
        )
    assert len(representation.quadratic_coefficients) == 14 * 13 // 2
    assert set(representation.quadratic_coefficients.values()) == {2.0 * penalty}
    for row in range(14):
        assert representation.upper_triangular_matrix[row][row] == pytest.approx(
            representation.scores[row] - penalty
        )
        assert all(
            representation.upper_triangular_matrix[row][column] == 0.0
            for column in range(row)
        )


def test_independent_energy_matches_qiskit_for_all_16384_states(
    reference_model,
) -> None:
    validation = validate_all_qubo_energies(reference_model, tolerance=1e-10)
    assert validation.states_checked == 2**14
    assert validation.maximum_absolute_difference <= 1e-10


def test_sufficient_penalty_makes_every_infeasible_state_more_expensive(
    reference_model,
) -> None:
    best_feasible_score = min(reference_model.representation.scores)
    for bits in product((0, 1), repeat=14):
        selected_count = sum(bits)
        score_sum = sum(
            score * bit
            for score, bit in zip(
                reference_model.representation.scores, bits, strict=True
            )
        )
        expected_penalty = (
            reference_model.representation.penalty * (selected_count - 1) ** 2
        )
        assert qubo_energy(reference_model, bits) == pytest.approx(
            score_sum + expected_penalty, abs=1e-10
        )
        if selected_count != 1:
            assert expected_penalty > 0.0
            assert qubo_energy(reference_model, bits) > best_feasible_score


def test_reference_classical_enumeration_and_exact_eigensolver_agree(
    cc2_alternatives, reference_model
) -> None:
    direct = solve_direct_argmin(cc2_alternatives, REFERENCE_WEIGHTS)
    exhaustive = solve_exhaustive_binary(cc2_alternatives, REFERENCE_WEIGHTS)
    assert_solver_agreement(direct, exhaustive)
    enumerated = solve_qubo_by_enumeration(reference_model)
    exact = solve_qubo_with_exact_eigensolver(reference_model)

    expected_id = direct.selected.alternative.alternative_id
    assert enumerated.selected_alternative is not None
    assert exact.selected_alternative is not None
    assert enumerated.selected_alternative.alternative_id == expected_id
    assert exact.selected_alternative.alternative_id == expected_id
    assert enumerated.energy == pytest.approx(direct.selected.weighted_score, abs=1e-10)
    assert exact.energy == pytest.approx(direct.selected.weighted_score, abs=1e-10)
    assert enumerated.original_weighted_score == pytest.approx(
        direct.selected.weighted_score
    )
    assert enumerated.feasible
    assert exact.feasible
    assert sum(enumerated.bit_values) == sum(exact.bit_values) == 1


def test_qiskit_and_documented_bit_orders_are_explicit(
    cc2_alternatives, reference_model
) -> None:
    direct = solve_direct_argmin(cc2_alternatives, REFERENCE_WEIGHTS)
    solution = solve_qubo_with_exact_eigensolver(reference_model)
    selected_index = reference_model.representation.alternative_ids.index(
        direct.selected.alternative.alternative_id
    )
    assert solution.bit_values[selected_index] == 1
    assert solution.bit_string == "".join(str(bit) for bit in solution.bit_values)
    # Qiskit state labels display x_(n-1)..x_0, opposite the result.x vector.
    assert solution.qiskit_state_label == solution.bit_string[::-1]


def test_qubo_is_converted_to_a_fourteen_qubit_ising_operator(
    reference_model,
) -> None:
    operator, offset = to_ising(reference_model)
    assert operator.num_qubits == 14
    assert isclose(float(offset), offset)


@pytest.mark.parametrize(
    "invalid_penalty", [0.0, -1.0, float("nan"), float("inf"), "invalid", True]
)
def test_invalid_penalties_are_rejected(cc2_alternatives, invalid_penalty) -> None:
    with pytest.raises((TypeError, ValueError), match="finite positive"):
        build_qubo_model(cc2_alternatives, REFERENCE_WEIGHTS, penalty=invalid_penalty)


def test_penalty_sensitivity_uses_actual_minimum_score(cc2_alternatives) -> None:
    baseline = build_qubo_model(cc2_alternatives, REFERENCE_WEIGHTS)
    threshold = minimum_safe_penalty_threshold(baseline.representation.scores)
    assert threshold == pytest.approx(min(baseline.representation.scores))

    too_small = threshold / 2.0
    unsafe_model = build_qubo_model(
        cc2_alternatives, REFERENCE_WEIGHTS, penalty=too_small
    )
    unsafe_solution = solve_qubo_by_enumeration(unsafe_model)
    assert not is_penalty_sufficient(unsafe_model.representation.scores, too_small)
    assert unsafe_solution.bit_values == (0,) * 14
    assert not unsafe_solution.feasible

    sufficient = threshold + 1e-6
    safe_model = build_qubo_model(
        cc2_alternatives, REFERENCE_WEIGHTS, penalty=sufficient
    )
    safe_solution = solve_qubo_by_enumeration(safe_model)
    assert is_penalty_sufficient(safe_model.representation.scores, sufficient)
    assert safe_solution.feasible
    assert is_penalty_sufficient(baseline.representation.scores, DEFAULT_QUBO_PENALTY)


@pytest.mark.parametrize("benchmark", [Benchmark.CC1, Benchmark.CC2])
@pytest.mark.parametrize(
    "weights",
    [
        ObjectiveWeights(0.8, 0.1, 0.1),
        ObjectiveWeights(0.1, 0.8, 0.1),
        ObjectiveWeights(0.1, 0.1, 0.8),
        REFERENCE_WEIGHTS,
    ],
    ids=["fee-focused", "fx-focused", "speed-focused", "reference"],
)
def test_cc1_cc2_and_multiple_weights_match_classical_optimum(
    request, benchmark, weights
) -> None:
    alternatives = request.getfixturevalue(
        "cc1_alternatives" if benchmark is Benchmark.CC1 else "cc2_alternatives"
    )
    direct = solve_direct_argmin(alternatives, weights)
    model = build_qubo_model(alternatives, weights, penalty=2.0)
    qubo = solve_qubo_by_enumeration(model)
    assert qubo.feasible
    assert qubo.original_weighted_score == pytest.approx(
        direct.selected.weighted_score, abs=1e-10
    )
    assert qubo.selected_alternative is not None
    assert qubo.selected_alternative.alternative_id in (
        direct.tie_information.tied_alternative_ids
    )


def test_reversed_input_produces_deterministic_model_and_result(
    cc2_alternatives, reference_model
) -> None:
    repeated = build_qubo_model(
        tuple(reversed(cc2_alternatives)), REFERENCE_WEIGHTS, penalty=2.0
    )
    assert repeated.representation == reference_model.representation
    assert (
        solve_qubo_by_enumeration(repeated).bit_values
        == solve_qubo_by_enumeration(reference_model).bit_values
    )


def test_feasible_energy_ties_use_existing_classical_tie_breaking(
    cc2_alternatives,
) -> None:
    speed_only = ObjectiveWeights(0.0, 0.0, 1.0)
    direct = solve_direct_argmin(cc2_alternatives, speed_only)
    assert direct.tie_information.tie_breaking_applied
    model = build_qubo_model(cc2_alternatives, speed_only, penalty=2.0)
    enumerated = solve_qubo_by_enumeration(model)
    assert enumerated.selected_alternative is not None
    assert (
        enumerated.selected_alternative.alternative_id
        == direct.selected.alternative.alternative_id
    )


def test_energy_rejects_wrong_length_and_nonbinary_values(reference_model) -> None:
    with pytest.raises(ValueError, match="14 binary"):
        qubo_energy(reference_model, (0, 1))
    with pytest.raises(ValueError, match="14 binary"):
        qubo_energy(reference_model, (0,) * 13 + (2,))
