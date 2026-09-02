"""Validation tests for W-state and ring-XY constraint-preserving QAOA."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from math import isclose

import pytest
from qiskit.quantum_info import Statevector

from backend.app.optimization.constraint_mixer import (
    CONSTRAINT_TOLERANCE,
    build_xy_ring_hamiltonian,
    build_xy_ring_mixer_circuit,
    excitation_number_operator,
    ring_connectivity,
    validate_constraint_preserving_design,
)
from backend.app.optimization.initial_states import (
    build_uniform_one_hot_state,
    one_hot_qiskit_label,
    validate_uniform_one_hot_state,
)
from backend.app.optimization.models import (
    Benchmark,
    ObjectiveWeights,
    ServiceAlternative,
)
from backend.app.optimization.qaoa_experiment import (
    REFERENCE_SEEDS,
    build_mixer_comparison,
    run_qaoa_experiment,
)
from backend.app.optimization.qaoa_models import QaoaConfig, QaoaMixer
from backend.app.optimization.qaoa_solver import (
    interpret_sample_distribution,
    run_qaoa,
)
from backend.app.optimization.qubo import build_qubo_model
from backend.scripts import run_qaoa as run_qaoa_script


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
    return build_qubo_model(
        tuple(
            _alternative(identifier, index)
            for index, identifier in enumerate(("alt-a", "alt-b", "alt-c"))
        ),
        ObjectiveWeights(0.4, 0.3, 0.3),
        penalty=2.0,
    )


@pytest.fixture(scope="module")
def xy_run(reduced_model):
    return run_qaoa(
        reduced_model,
        QaoaConfig(
            mixer=QaoaMixer.CONSTRAINT_PRESERVING_XY,
            shots=64,
            max_iterations=4,
            algorithm_seed=11,
        ),
    )


def test_ring_connectivity_is_periodic_and_deterministic() -> None:
    assert ring_connectivity(4) == ((0, 1), (1, 2), (2, 3), (3, 0))
    assert ring_connectivity(14)[-1] == (13, 0)
    assert len(ring_connectivity(14)) == 14


def test_uniform_w_state_is_normalized_equal_and_feasible() -> None:
    validation = validate_uniform_one_hot_state(14)
    assert validation.norm == pytest.approx(1.0)
    assert validation.expected_one_hot_probability == pytest.approx(1.0 / 14.0)
    assert validation.maximum_probability_error <= 1e-14
    assert validation.infeasible_probability == 0.0
    assert len(validation.one_hot_probabilities_by_qiskit_label) == 14


def test_w_state_variable_to_state_label_mapping() -> None:
    state = Statevector.from_instruction(build_uniform_one_hot_state(14))
    assert one_hot_qiskit_label(14, 0) == "00000000000001"
    assert one_hot_qiskit_label(14, 13) == "10000000000000"
    for index in range(14):
        label = one_hot_qiskit_label(14, index)
        amplitude = state.data[int(label, 2)]
        assert amplitude.real == pytest.approx(1.0 / (14**0.5), abs=1e-14)
        assert amplitude.imag == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("qubit_count", [3, 14])
def test_xy_hamiltonian_is_hermitian_and_commutes_with_number(qubit_count) -> None:
    mixer = build_xy_ring_hamiltonian(qubit_count)
    hermitian_difference = (mixer - mixer.adjoint()).simplify(atol=1e-12)
    number = excitation_number_operator(qubit_count)
    commutator = (mixer.compose(number) - number.compose(mixer)).simplify(atol=1e-12)
    assert max(abs(value) for value in hermitian_difference.coeffs) <= 1e-12
    assert max(abs(value) for value in commutator.coeffs) <= 1e-12


@pytest.mark.parametrize("qubit_count", [3, 14])
def test_xy_mixer_preserves_hamming_weight_from_w_state(qubit_count) -> None:
    validation = validate_constraint_preserving_design(qubit_count)
    assert validation.maximum_mixer_leakage <= CONSTRAINT_TOLERANCE
    assert all(
        leakage <= CONSTRAINT_TOLERANCE
        for leakage in validation.mixer_leakage_by_angle.values()
    )


@pytest.mark.parametrize("qubit_count", [3, 14])
def test_xy_mixer_preserves_a_single_one_hot_state(qubit_count) -> None:
    state = Statevector.from_label(one_hot_qiskit_label(qubit_count, 0))
    mixer = build_xy_ring_mixer_circuit(qubit_count)
    beta = next(iter(mixer.parameters))
    for angle in (0.11, 0.47, 1.03):
        evolved = state.evolve(mixer.assign_parameters({beta: angle}))
        leakage = sum(
            probability
            for label, probability in evolved.probabilities_dict().items()
            if label.count("1") != 1
        )
        assert leakage <= CONSTRAINT_TOLERANCE


def test_leakage_is_probability_outside_hamming_weight_one(reduced_model) -> None:
    interpreted = interpret_sample_distribution(
        reduced_model,
        {"000": 0.2, "001": 0.3, "010": 0.1, "111": 0.4},
    )
    assert interpreted["hamming_weight_distribution"] == pytest.approx(
        {"0": 0.2, "1": 0.4, "2": 0.0, "3": 0.4}
    )
    assert interpreted["feasible_probability"] == pytest.approx(0.4)
    assert interpreted["leakage_probability"] == pytest.approx(0.6)


def test_invalid_mixer_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="mixer must be one of"):
        QaoaConfig(mixer="not_a_mixer")


def test_constraint_preserving_run_has_no_leakage_and_serializes(xy_run) -> None:
    assert xy_run.metrics.mixer_type == "constraint_preserving_xy"
    assert xy_run.metrics.initial_state_type == "uniform_one_hot_w_state"
    assert xy_run.metrics.mixer_connectivity == ((0, 1), (1, 2), (2, 0))
    assert xy_run.feasible_probability == pytest.approx(1.0)
    assert xy_run.constraint_leakage_probability == pytest.approx(0.0)
    assert xy_run.raw_most_probable_state.feasible
    payload = json.loads(json.dumps(asdict(xy_run), allow_nan=False))
    assert payload["hamming_weight_distribution"]["1"] == pytest.approx(1.0)


def test_comparison_aggregation_supports_stored_standard_and_xy_results(
    tmp_path, xy_run
) -> None:
    run_payload = asdict(xy_run)
    common_config = {
        "period": "2025_3Q",
        "corridor": "KENTZA",
        "benchmark": "CC2",
        "weights": {"fee_weight": 0.4, "fx_weight": 0.3, "speed_weight": 0.3},
        "penalty": 2.0,
        "benchmark_amount": 100.0,
        "benchmark_currency": "KES",
        "minimum_safe_penalty_condition": "P > 1.0",
        "qubit_count": 3,
        "shots": 64,
        "optimizer": "COBYLA",
        "maximum_iterations": 4,
        "simulator": "test sampler",
        "reps": 1,
        "seeds": [11],
        "provenance": {
            "processed_dataset_sha256": "test-hash",
            "normalization": {"method": "test"},
            "objective_inputs_in_variable_order": ["alt-a", "alt-b", "alt-c"],
        },
    }
    exact_reference = {
        "bit_string_x0_to_xn": "100",
        "qiskit_state_label_xn_to_x0": "001",
        "qubo_energy": 0.0,
        "weighted_score": 0.0,
        "feasible": True,
        "alternative_id": "alt-a",
        "provider": "Provider 0",
        "payment_instrument": "Cash",
        "pickup_method": "Cash",
    }
    package_versions = {"python": "test"}
    standard_run = json.loads(json.dumps(run_payload))
    standard_run.pop("constraint_leakage_probability")
    standard_run["feasible_probability"] = 0.25
    standard_run["raw_most_probable_state"]["feasible"] = False
    standard_payload = {
        "schema_version": "1.0",
        "experiment_configuration": {**common_config, "mixer": "standard X mixer"},
        "package_versions": package_versions,
        "exact_reference": exact_reference,
        "runs": [standard_run],
    }
    xy_payload = {
        "schema_version": "2.0",
        "experiment_configuration": {
            **common_config,
            "mixer": "constraint_preserving_xy",
            "initial_state": "uniform_one_hot_w_state",
        },
        "package_versions": package_versions,
        "exact_reference": exact_reference,
        "runs": [run_payload],
    }
    standard_path = tmp_path / "standard.json"
    xy_path = tmp_path / "xy.json"
    standard_path.write_text(json.dumps(standard_payload), encoding="utf-8")
    xy_path.write_text(json.dumps(xy_payload), encoding="utf-8")
    comparison = build_mixer_comparison(
        {"standard_x_p1": standard_path, "constraint_preserving_xy_p1": xy_path}
    )
    entries = {item["name"]: item for item in comparison["experiments"]}
    assert entries["standard_x_p1"]["constraint_leakage_probability"][
        "mean"
    ] == pytest.approx(0.75)
    assert entries["constraint_preserving_xy_p1"]["feasible_probability"][
        "mean"
    ] == pytest.approx(1.0)
    assert entries["constraint_preserving_xy_p1"]["modal_state_feasibility_rate"] == 1.0
    assert comparison["schema_version"] == "2.0"
    assert comparison["comparison_basis"]["input_schema_versions"] == {
        "standard_x_p1": "1.0",
        "constraint_preserving_xy_p1": "2.0",
    }
    assert comparison["comparison_basis"]["objective_inputs_in_variable_order"] == [
        "alt-a",
        "alt-b",
        "alt-c",
    ]
    with pytest.raises(ValueError, match="required mixer and depth"):
        build_mixer_comparison({"standard_x_p1": xy_path})


@pytest.mark.parametrize(
    ("section", "key", "replacement", "message"),
    [
        ("configuration", "shots", 128, "execution protocol"),
        ("configuration", "benchmark_amount", 200.0, "problem instance"),
        ("configuration", "seeds", [29], "recorded seeds"),
        (
            "provenance",
            "objective_inputs_in_variable_order",
            ["different"],
            "objective inputs",
        ),
    ],
)
def test_comparison_rejects_materially_incompatible_experiments(
    tmp_path, xy_run, section, key, replacement, message
) -> None:
    run_payload = json.loads(json.dumps(asdict(xy_run)))
    configuration = {
        "period": "test_period",
        "corridor": "TEST",
        "benchmark": "CC2",
        "benchmark_amount": 100.0,
        "benchmark_currency": "KES",
        "weights": {"fee_weight": 0.4, "fx_weight": 0.3, "speed_weight": 0.3},
        "penalty": 2.0,
        "minimum_safe_penalty_condition": "P > 1.0",
        "qubit_count": 3,
        "shots": 64,
        "optimizer": "COBYLA",
        "maximum_iterations": 4,
        "simulator": "test sampler",
        "reps": 1,
        "seeds": [11],
        "mixer": "constraint_preserving_xy",
        "provenance": {
            "processed_dataset_sha256": "test-hash",
            "normalization": {"method": "test"},
            "objective_inputs_in_variable_order": ["alt-a", "alt-b", "alt-c"],
        },
    }
    exact_reference = {
        "bit_string_x0_to_xn": "100",
        "qiskit_state_label_xn_to_x0": "001",
        "qubo_energy": 0.0,
        "weighted_score": 0.0,
        "feasible": True,
        "alternative_id": "alt-a",
        "provider": "Provider 0",
        "payment_instrument": "Cash",
        "pickup_method": "Cash",
    }
    first = {
        "schema_version": "2.0",
        "experiment_configuration": configuration,
        "package_versions": {"python": "test"},
        "exact_reference": exact_reference,
        "runs": [run_payload],
    }
    second = json.loads(json.dumps(first))
    target = (
        second["experiment_configuration"]
        if section == "configuration"
        else second["experiment_configuration"]["provenance"]
    )
    target[key] = replacement
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        build_mixer_comparison({"first": first_path, "second": second_path})


def test_run_cli_checks_overwrite_before_starting_experiment(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run_qaoa", "--output", str(output)])

    def fail_if_called(**_kwargs):
        raise AssertionError(
            "QAOA must not run when overwrite protection rejects output"
        )

    monkeypatch.setattr(run_qaoa_script, "run_qaoa_experiment", fail_if_called)
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        run_qaoa_script.main()
    assert output.read_text(encoding="utf-8") == "preserve me"


def test_comparison_rejects_unsupported_result_schema(tmp_path, xy_run) -> None:
    payload = {
        "schema_version": "99.0",
        "experiment_configuration": {},
        "package_versions": {},
        "exact_reference": {},
        "runs": [asdict(xy_run)],
    }
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported experiment schema"):
        build_mixer_comparison({"unsupported": path})


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("reps", [1, 2])
def test_full_five_seed_constraint_preserving_experiment(reps) -> None:
    result = run_qaoa_experiment(
        mixer=QaoaMixer.CONSTRAINT_PRESERVING_XY,
        reps=reps,
        seeds=REFERENCE_SEEDS,
    )
    assert len(result.runs) == 5
    assert all(
        isclose(run.feasible_probability, 1.0, rel_tol=0.0, abs_tol=1e-12)
        for run in result.runs
    )
    assert all(run.constraint_leakage_probability <= 1e-12 for run in result.runs)
