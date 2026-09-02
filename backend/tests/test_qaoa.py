"""Fast deterministic tests for QAOA decoding, metrics, and serialization."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from backend.app.optimization.classical_solver import solve_direct_argmin
from backend.app.optimization.mathematical_solver import (
    solve_constrained_with_scipy_milp,
)
from backend.app.optimization.models import (
    Benchmark,
    ObjectiveWeights,
    ServiceAlternative,
)
from backend.app.optimization.qaoa_experiment import (
    REFERENCE_SEEDS,
    aggregate_qaoa_runs,
    assert_mathematical_baseline_agreement,
    run_qaoa_experiment,
)
from backend.app.optimization.qaoa_models import QaoaConfig
from backend.app.optimization.qaoa_solver import (
    interpret_sample_distribution,
    qiskit_label_to_x_order,
    run_qaoa,
    x_order_to_qiskit_label,
)
from backend.app.optimization.qubo import build_qubo_model


def _alternative(identifier: str, index: int) -> ServiceAlternative:
    return ServiceAlternative(
        alternative_id=identifier,
        period="test_period",
        corridor="TEST",
        firm=f"Provider {index}",
        firm_type="Test",
        payment_instrument="Cash",
        speed_label=("Less than one hour", "Same day", "Next day")[index],
        speed_ordinal=index,
        pickup_method="Cash",
        benchmark=Benchmark.CC2,
        benchmark_amount=100.0,
        benchmark_currency="KES",
        fee_percentage=float(index),
        fx_margin=float(index),
        total_cost_percentage=float(index * 2),
        total_cost_residual=0.0,
    )


@pytest.fixture(scope="module")
def reduced_model():
    alternatives = tuple(
        _alternative(identifier, index)
        for index, identifier in enumerate(("alt-a", "alt-b", "alt-c"))
    )
    return build_qubo_model(
        alternatives,
        ObjectiveWeights(0.4, 0.3, 0.3),
        penalty=2.0,
    )


@pytest.fixture(scope="module")
def seeded_runs(reduced_model):
    config = QaoaConfig(shots=128, max_iterations=4, algorithm_seed=17)
    return run_qaoa(reduced_model, config), run_qaoa(reduced_model, config)


def test_bit_order_conversion_is_explicit_and_reversible() -> None:
    assert qiskit_label_to_x_order("001", 3) == (1, 0, 0)
    assert x_order_to_qiskit_label((1, 0, 0)) == "001"
    with pytest.raises(ValueError, match="3-bit"):
        qiskit_label_to_x_order("01", 3)


def test_distribution_keeps_infeasible_raw_mode_and_finds_best_feasible(
    reduced_model,
) -> None:
    interpreted = interpret_sample_distribution(
        reduced_model,
        {"000": 0.5, "010": 0.3, "001": 0.2},
    )
    assert not interpreted["raw"].feasible
    assert interpreted["raw"].qiskit_state_label_xn_to_x0 == "000"
    assert interpreted["best_feasible"].bit_string_x0_to_xn == "100"
    assert interpreted["feasible_probability"] == pytest.approx(0.5)
    assert interpreted["optimal_probability"] == pytest.approx(0.2)
    assert interpreted["exact_sampled"]
    assert not interpreted["exact_most_probable"]
    assert interpreted["gap"] == pytest.approx(0.0)


def test_no_feasible_sample_is_structured_without_recommendation(reduced_model) -> None:
    interpreted = interpret_sample_distribution(
        reduced_model,
        {"000": 0.75, "111": 0.25},
    )
    assert interpreted["best_feasible"] is None
    assert interpreted["best_score"] is None
    assert interpreted["gap"] is None
    assert interpreted["feasible_probability"] == 0.0
    assert interpreted["optimal_probability"] == 0.0


def test_suboptimal_best_feasible_gap_is_absolute(reduced_model) -> None:
    interpreted = interpret_sample_distribution(reduced_model, {"010": 1.0})
    assert interpreted["best_feasible"].selected_alternative_id == "alt-b"
    assert interpreted["best_score"] == pytest.approx(0.5)
    assert interpreted["gap"] == pytest.approx(0.5)
    assert not interpreted["exact_sampled"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reps": 0},
        {"shots": 0},
        {"max_iterations": 0},
        {"algorithm_seed": -1},
        {"optimizer": "SLSQP"},
        {"transpiler_optimization_level": 4},
    ],
)
def test_invalid_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        QaoaConfig(**kwargs)


def test_qaoa_consumes_prebuilt_qubo_and_records_required_metrics(
    reduced_model, seeded_runs
) -> None:
    first, _ = seeded_runs
    assert reduced_model.representation.alternative_ids == ("alt-a", "alt-b", "alt-c")
    assert first.metrics.qubit_count == 3
    assert first.metrics.qaoa_repetitions == 1
    assert first.metrics.shots == 128
    assert first.metrics.actual_evaluations == len(first.callback_history)
    assert first.metrics.circuit.number_of_parameters == 2
    assert first.metrics.circuit.original_depth > 0
    assert first.metrics.circuit.decomposed_depth > 0
    assert first.metrics.circuit.transpiled_depth > 0
    assert first.metrics.ising_conversion_time_seconds >= 0.0
    assert first.metrics.circuit_construction_time_seconds >= 0.0
    assert first.metrics.transpilation_time_seconds >= 0.0
    assert first.metrics.optimizer_sampler_time_seconds >= 0.0
    assert first.metrics.final_sampling_time_seconds >= 0.0
    assert first.metrics.queue_time_seconds == 0.0
    assert (
        first.raw_most_probable_state.bit_string_x0_to_xn
        == (first.raw_most_probable_state.qiskit_state_label_xn_to_x0[::-1])
    )


def test_seeded_qaoa_is_deterministic_where_supported(seeded_runs) -> None:
    first, second = seeded_runs
    assert first.optimal_parameters == pytest.approx(second.optimal_parameters)
    assert first.feasible_state_probabilities == second.feasible_state_probabilities
    assert first.raw_most_probable_state == second.raw_most_probable_state
    assert first.callback_history == second.callback_history


def test_result_serialization_is_valid_json(seeded_runs) -> None:
    first, _ = seeded_runs
    payload = json.loads(json.dumps(first, default=lambda value: value.__dict__))
    assert payload["metrics"]["optimizer"] == "COBYLA"
    assert "approximation_ratio" not in payload


def test_mathematical_baseline_agrees_on_reduced_problem(reduced_model) -> None:
    result = solve_constrained_with_scipy_milp(reduced_model)
    assert result.success
    assert result.feasible
    assert result.status == "SUCCESS"
    assert result.bit_values == (1, 0, 0)
    assert result.selected_alternative.alternative_id == "alt-a"
    assert result.weighted_score == pytest.approx(0.0)
    assert result.mip_gap == pytest.approx(0.0)


def test_mathematical_baseline_accepts_a_different_tied_optimum(reduced_model) -> None:
    tied_alternatives = (
        reduced_model.alternatives[0],
        replace(reduced_model.alternatives[1], speed_ordinal=0),
        reduced_model.alternatives[2],
    )
    tied_model = build_qubo_model(
        tied_alternatives,
        ObjectiveWeights(0.0, 0.0, 1.0),
        penalty=2.0,
    )
    mathematical = solve_constrained_with_scipy_milp(tied_model)
    mathematical = replace(
        mathematical,
        bit_values=(0, 1, 0),
        selected_alternative=tied_model.alternatives[1],
        weighted_score=0.0,
    )
    classical = solve_direct_argmin(tied_alternatives, ObjectiveWeights(0.0, 0.0, 1.0))
    assert mathematical.selected_alternative.alternative_id != (
        classical.selected.alternative.alternative_id
    )
    assert_mathematical_baseline_agreement(classical, mathematical)


def test_experiment_rejects_penalty_that_changes_feasible_target() -> None:
    with pytest.raises(ValueError, match="requires P >"):
        run_qaoa_experiment(
            penalty=0.0001,
            shots=16,
            max_iterations=4,
            seeds=(11,),
        )


def test_aggregate_calculations_include_failures_and_population_stddev(
    seeded_runs,
) -> None:
    first, _ = seeded_runs
    recovered = replace(
        first,
        success=True,
        exact_optimum_sampled=True,
        optimal_state_probability=0.4,
        feasible_probability=0.6,
        absolute_optimality_gap=0.0,
        metrics=replace(first.metrics, total_execution_time_seconds=2.0),
    )
    unsuccessful = replace(
        first,
        success=False,
        exact_optimum_sampled=False,
        optimal_state_probability=0.0,
        feasible_probability=0.0,
        absolute_optimality_gap=None,
        metrics=replace(first.metrics, total_execution_time_seconds=4.0),
    )
    aggregate = aggregate_qaoa_runs((recovered, unsuccessful))
    assert aggregate.successful_feasible_run_count == 1
    assert aggregate.exact_optimum_recovery_count == 1
    assert aggregate.exact_optimum_recovery_rate == 0.5
    assert aggregate.optimal_state_probability.mean == pytest.approx(0.2)
    assert aggregate.optimal_state_probability.standard_deviation == pytest.approx(0.2)
    assert aggregate.feasible_probability.mean == pytest.approx(0.3)
    assert aggregate.best_feasible_gap.mean == 0.0
    assert aggregate.runtime_seconds.mean == 3.0
    assert aggregate.runtime_seconds.standard_deviation == 1.0


@pytest.mark.slow
@pytest.mark.integration
def test_full_five_seed_reference_experiment() -> None:
    result = run_qaoa_experiment(seeds=REFERENCE_SEEDS)
    assert len(result.runs) == 5
    assert result.experiment_configuration["qubit_count"] == 14
    assert result.exact_reference["alternative_id"] == "alt-cc1e0457fb2b"
    assert all(
        run.best_feasible_weighted_score is None
        or run.best_feasible_weighted_score >= result.exact_reference["weighted_score"]
        for run in result.runs
    )
