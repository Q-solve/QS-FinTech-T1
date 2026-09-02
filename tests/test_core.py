from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from qkash.benchmark import rank_metrics, validate_solver_outputs
from qkash.batch import BatchSettings, build_batch_plans
from qkash.classical import simulated_annealing
from qkash.constraints import PolicyConstraints, apply_policy_constraints
from qkash.data import (
    FilterSpec,
    ensure_supported_columns,
    filter_dataset,
    latest_service_options,
    match_transfer_amount,
    parse_dates,
    prepare_candidates,
    settlement_days,
)
from qkash.profiling import infer_use_case_policy
from qkash.quantum import (
    DEFAULT_MAX_QUBITS,
    LocalAerBackend,
    QBraidBackend,
    QuantumBackend,
    run_qaoa,
)
from qkash.scoring import (
    ENCODING_BINARY_INDEX,
    ENCODING_ONE_HOT,
    add_objective_losses,
    build_selection_qubo,
    pareto_prune,
    required_qubits,
    score_candidates,
    select_qubo_candidates,
    verify_qubo_equivalence,
)


def sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "period": "2025_1Q",
                "source_name": "Kenya",
                "destination_name": "Uganda",
                "firm": "FastCash",
                "firm_type": "Mobile Operator",
                "access point": "Mobile phone",
                "speed actual": "Less than one hour",
                "cc1 lcu amount": 2000.0,
                "cc1 denomination amount": 200.0,
                "cc1 lcu fee": 20.0,
                "cc1 fx margin": 0.2,
                "cc1 total cost %": 1.2,
                "cc2 lcu amount": 5000.0,
                "cc2 denomination amount": 500.0,
                "cc2 lcu fee": 40.0,
                "cc2 fx margin": 0.2,
                "cc2 total cost %": 1.0,
                "transparent": "yes",
                "receiving network coverage": "Nationwide",
                "pickup method": "Cash",
                "date": "01/Jan/2025",
                "corridor": "KENUGA",
                "payment instrument": "Mobile money",
            },
            {
                "id": 2,
                "period": "2025_1Q",
                "source_name": "Kenya",
                "destination_name": "Uganda",
                "firm": "SlowBank",
                "firm_type": "Bank",
                "access point": "Bank branch",
                "speed actual": "3-5 days",
                "cc1 lcu amount": 2000.0,
                "cc1 denomination amount": 200.0,
                "cc1 lcu fee": 60.0,
                "cc1 fx margin": 1.0,
                "cc1 total cost %": 4.5,
                "cc2 lcu amount": 5000.0,
                "cc2 denomination amount": 500.0,
                "cc2 lcu fee": 120.0,
                "cc2 fx margin": 1.0,
                "cc2 total cost %": 4.0,
                "transparent": "no",
                "receiving network coverage": "Main city",
                "pickup method": "Bank Account",
                "date": "01/Jan/2025",
                "corridor": "KENUGA",
                "payment instrument": "Bank account transfer",
            },
            {
                "id": 3,
                "period": "2025_1Q",
                "source_name": "Rwanda",
                "destination_name": "Kenya",
                "firm": "OutsideDominant",
                "firm_type": "Mobile Operator",
                "access point": "Mobile phone",
                "speed actual": "Less than one hour",
                "cc1 lcu amount": 2000.0,
                "cc1 denomination amount": 200.0,
                "cc1 lcu fee": 1.0,
                "cc1 fx margin": 0.01,
                "cc1 total cost %": 0.1,
                "cc2 lcu amount": 5000.0,
                "cc2 denomination amount": 500.0,
                "cc2 lcu fee": 2.0,
                "cc2 fx margin": 0.01,
                "cc2 total cost %": 0.1,
                "transparent": "yes",
                "receiving network coverage": "Nationwide",
                "pickup method": "Cash",
                "date": "01/Jan/2025",
                "corridor": "RWAKEN",
                "payment instrument": "Mobile money",
            },
        ]
    ).assign(
        date_parsed=lambda frame: pd.to_datetime(frame["date"], format="%d/%b/%Y"),
        period_order=20251,
    )


def test_filtering_happens_before_pareto_pruning() -> None:
    df = sample_dataframe()
    filtered = filter_dataset(df, FilterSpec(source_name="Kenya", destination_name="Uganda"))
    prepared = prepare_candidates(filtered, amount_tier="cc1")
    scored = score_candidates(prepared, {"transaction_fee": 1.0})
    pruned = pareto_prune(scored)

    assert set(pruned["source_name"]) == {"Kenya"}
    assert "OutsideDominant" not in set(pruned["firm"])


def test_filter_spec_uses_single_optional_values() -> None:
    df = sample_dataframe()
    filtered = filter_dataset(
        df,
        FilterSpec(
            source_name="Kenya",
            destination_name="Uganda",
            firm_type="Mobile Operator",
            pickup_method="Cash",
            payment_instrument="Mobile money",
            access_point="Mobile phone",
        ),
    )

    assert filtered["firm"].tolist() == ["FastCash"]


def test_missing_optional_access_point_column_does_not_crash() -> None:
    df = sample_dataframe().drop(columns=["access point"])
    filtered = filter_dataset(df, FilterSpec(access_point="Mobile phone"))

    assert filtered.empty


def test_qubo_matches_original_constrained_model() -> None:
    qubo = build_selection_qubo([0.3, 0.1, 0.7], penalty=2.0)
    verification = verify_qubo_equivalence(qubo)

    assert verification["equivalent"] is True
    assert verification["original_indices"] == [1]
    assert verification["qubo_indices"] == [1]


def test_simulated_annealing_produces_samples() -> None:
    qubo = build_selection_qubo([0.3, 0.1, 0.7], penalty=2.0)
    result = simulated_annealing(qubo, num_reads=8, sweeps=50, seed=7)

    assert len(result["samples"]) == 8
    assert any(sample["feasible"] for sample in result["samples"])


def test_qaoa_returns_circuit_diagram() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    qubo = build_selection_qubo([0.1, 0.3], penalty=2.0)
    result = run_qaoa(
        qubo,
        reps=1,
        shots=8,
        max_qubits=2,
        seed=11,
        optimizer_maxiter=4,
        backend_name="local_aer",
    )

    assert result["status"] == "ok"
    assert result["optimization_backend"] == "LocalAerBackend"
    assert result["execution_backend"] == "LocalAerBackend"
    assert result["circuit_num_qubits"] == 2
    assert result["candidate_count"] == 2
    assert result["basis_state_count"] == 4
    assert result["circuit_depth"] > 0
    assert len(result["samples"]) == 8
    assert "circuit_diagram" in result
    assert "Rz" in result["circuit_diagram"]


def test_qaoa_circuit_iterations_aggregate_samples() -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    qubo = build_selection_qubo([0.1, 0.3], penalty=2.0)
    result = run_qaoa(
        qubo,
        reps=1,
        shots=4,
        max_qubits=2,
        seed=11,
        optimizer_maxiter=4,
        execution_iterations=3,
        backend_name="local_aer",
    )

    assert result["status"] == "ok"
    assert result["execution_iterations"] == 3
    assert result["shots_per_iteration"] == 4
    assert result["total_shots"] == 12
    assert sum(result["measurement_counts"].values()) == 12
    assert len(result["samples"]) == 12


def test_quantum_backend_architecture() -> None:
    assert issubclass(LocalAerBackend, QuantumBackend)
    assert issubclass(QBraidBackend, QuantumBackend)


def test_solver_output_validation_rejects_bad_samples() -> None:
    validations = validate_solver_outputs(
        [
            {
                "algorithm": "bad",
                "samples": [
                    {
                        "bits": [1, 1],
                        "index": 0,
                        "objective": 0.1,
                        "feasible": True,
                    }
                ],
            }
        ],
        [0.1, 0.2],
    )

    assert validations[0]["valid"] is False


def test_metric_ranking_uses_runtime_as_quality_tie_breaker() -> None:
    metrics = pd.DataFrame(
        [
            {
                "algorithm": "Business as usual",
                "feasibility_rate": 1.0,
                "objective_value": 0.0,
                "relative_optimality_gap": 0.0,
                "optimum_hit_probability": 1.0,
                "end_to_end_runtime_s": 0.0018,
                "stability": 1.0,
                "time_to_solution_s": 0.0018,
                "runs": 1,
            },
            {
                "algorithm": "Exact mathematical baseline",
                "feasibility_rate": 1.0,
                "objective_value": 0.0,
                "relative_optimality_gap": 0.0,
                "optimum_hit_probability": 1.0,
                "end_to_end_runtime_s": 0.00006,
                "stability": 1.0,
                "time_to_solution_s": 0.00006,
                "runs": 1,
            },
        ]
    )

    assert rank_metrics(metrics).iloc[0]["algorithm"] == "Exact mathematical baseline"


def test_metric_ranking_prefers_compute_time_over_wall_clock() -> None:
    """A queued remote backend must not be ranked on how busy the provider was."""

    metrics = pd.DataFrame(
        [
            {
                "algorithm": "Simulated annealing",
                "feasibility_rate": 1.0,
                "objective_value": 0.05,
                "relative_optimality_gap": 0.0,
                "optimum_hit_probability": 0.5,
                "end_to_end_runtime_s": 1.0,
                "compute_runtime_s": 1.0,
                "stability": 1.0,
                "time_to_solution_s": 1.0,
                "compute_time_to_solution_s": 1.0,
                "runs": 100,
            },
            {
                # Same quality, but 90 s of it was queue waiting.
                "algorithm": "QAOA",
                "feasibility_rate": 1.0,
                "objective_value": 0.05,
                "relative_optimality_gap": 0.0,
                "optimum_hit_probability": 0.5,
                "end_to_end_runtime_s": 90.5,
                "compute_runtime_s": 0.5,
                "stability": 1.0,
                "time_to_solution_s": 90.5,
                "compute_time_to_solution_s": 0.5,
                "runs": 100,
            },
        ]
    )

    assert rank_metrics(metrics).iloc[0]["algorithm"] == "QAOA"


def test_metric_ranking_falls_back_when_compute_columns_are_absent() -> None:
    """Older metric frames without the compute clock must still order correctly."""

    metrics = pd.DataFrame(
        [
            {"algorithm": "slow", "feasibility_rate": 1.0, "objective_value": 0.0,
             "relative_optimality_gap": 0.0, "optimum_hit_probability": 1.0,
             "end_to_end_runtime_s": 5.0, "stability": 1.0,
             "time_to_solution_s": 5.0, "runs": 1},
            {"algorithm": "fast", "feasibility_rate": 1.0, "objective_value": 0.0,
             "relative_optimality_gap": 0.0, "optimum_hit_probability": 1.0,
             "end_to_end_runtime_s": 0.1, "stability": 1.0,
             "time_to_solution_s": 0.1, "runs": 1},
        ]
    )

    assert rank_metrics(metrics).iloc[0]["algorithm"] == "fast"


def test_qaoa_default_max_qubits_is_twenty() -> None:
    assert DEFAULT_MAX_QUBITS == 20


def test_nine_candidates_use_nine_one_hot_qubits() -> None:
    qubo = build_selection_qubo([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    verification = verify_qubo_equivalence(qubo)

    assert required_qubits(9) == 9
    assert qubo.encoding == ENCODING_ONE_HOT
    assert qubo.size == 9
    assert qubo.candidate_count == 9
    assert qubo.state_count == 512
    assert verification["equivalent"] is True
    assert verification["qubo_indices"] == [8]


def test_qubo_encodes_every_candidate_score_not_just_the_optimum() -> None:
    """Regression: the QUBO must not be built around a precomputed answer.

    Two score vectors sharing an argmin but differing elsewhere must produce
    different Hamiltonians. A construction that plants the classical optimum
    yields identical matrices here, which makes any solver comparison circular.
    """

    shared_argmin = build_selection_qubo([0.10, 0.50, 0.90, 0.95], penalty=2.0)
    different_tail = build_selection_qubo([0.10, 0.11, 0.12, 0.13], penalty=2.0)

    assert not np.array_equal(shared_argmin.matrix, different_tail.matrix)
    assert np.allclose(np.diag(shared_argmin.matrix), np.array([0.10, 0.50, 0.90, 0.95]) - 2.0)


def test_qubo_energy_ordering_matches_score_ordering() -> None:
    """Feasible-state energies must rank candidates the way their scores do."""

    scores = [0.01, 0.02, 0.03, 0.04, 0.90, 0.91, 0.92, 0.93]
    qubo = build_selection_qubo(scores, penalty=2.0)

    ranked = sorted(
        (qubo.energy(bits), index)
        for bits, index in (
            (b, qubo.decode_index(b)) for b in itertools.product((0, 1), repeat=qubo.size)
        )
        if index is not None
    )

    assert [index for _energy, index in ranked] == list(np.argsort(scores))


def test_qubo_verification_fails_when_penalty_is_too_small() -> None:
    """The equivalence check must be falsifiable, not a tautology."""

    assert verify_qubo_equivalence(build_selection_qubo([0.5, 0.9], penalty=2.0))["equivalent"]
    assert not verify_qubo_equivalence(
        build_selection_qubo([0.5, 0.9], penalty=0.4)
    )["equivalent"]


def test_binary_index_encoding_is_rejected() -> None:
    """A quadratic form cannot represent an arbitrary score vector over index bits."""

    with pytest.raises(NotImplementedError):
        build_selection_qubo([0.1, 0.2, 0.3], penalty=2.0, encoding=ENCODING_BINARY_INDEX)


def test_qubo_candidate_selection_uses_pareto_frontier_by_default() -> None:
    scored = pd.DataFrame(
        {
            "weighted_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "transaction_fee_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
            "time_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
            "fx_spread_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )

    model_candidates, pruned = select_qubo_candidates(scored, target_size=5)

    assert len(pruned) == 1
    assert len(model_candidates) == 1
    assert model_candidates["model_source"].tolist().count("Pareto frontier") == 1


def test_qubo_candidate_selection_can_fill_for_fixed_qubit_experiments() -> None:
    scored = pd.DataFrame(
        {
            "weighted_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "transaction_fee_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
            "time_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
            "fx_spread_loss": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )

    model_candidates, pruned = select_qubo_candidates(
        scored,
        target_size=5,
        allow_fallback=True,
    )

    assert len(pruned) == 1
    assert len(model_candidates) == 5
    assert model_candidates["model_source"].tolist().count("Pareto frontier") == 1
    assert model_candidates["model_source"].tolist().count("Best scored fallback") == 4


def test_transfer_amount_matches_nearest_supported_tier() -> None:
    match = match_transfer_amount(sample_dataframe(), 450.0)

    assert match.amount_tier == "cc2"
    assert match.matched_denomination == 500.0


def test_latest_service_options_keeps_provider_method_latest_row() -> None:
    older = sample_dataframe().iloc[[0]].copy()
    newer = older.copy()
    newer["date"] = "2025-03-01"
    newer["period"] = "2025_1Q"
    newer["date_parsed"] = pd.Timestamp("2025-03-01")
    newer["period_order"] = 20251
    frame = pd.concat([older, newer], ignore_index=True)

    latest = latest_service_options(frame)

    assert len(latest) == 1
    assert latest.iloc[0]["date_parsed"] == pd.Timestamp("2025-03-01")


def test_profile_policy_is_inferred_without_user_weights() -> None:
    prepared = prepare_candidates(
        filter_dataset(sample_dataframe(), FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)
    profiled, inference = infer_use_case_policy(losses, transfer_amount=200.0, random_state=7)

    assert "service_profile_label" in profiled.columns
    assert set(inference.weights) == {"transaction_fee", "time", "fx_spread", "risk"}
    assert sum(inference.weights.values()) == pytest.approx(1.0)


def test_objective_losses_include_risk_loss() -> None:
    prepared = prepare_candidates(
        filter_dataset(sample_dataframe(), FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)

    assert "risk_loss" in losses.columns
    assert "coverage_loss" in losses.columns
    assert "transparency_loss" in losses.columns
    assert "access_point_loss" in losses.columns
    assert losses["risk_loss"].between(0.0, 1.0).all()


def test_missing_optional_risk_fields_do_not_create_synthetic_risk() -> None:
    source = sample_dataframe().drop(
        columns=["receiving network coverage", "access point"],
    )
    prepared = prepare_candidates(
        filter_dataset(source, FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)

    assert losses["coverage_loss"].eq(0.0).all()
    assert losses["access_point_loss"].eq(0.0).all()
    assert losses["risk_component_count"].eq(1).all()
    assert losses["risk_components"].eq("transparency").all()
    assert losses["risk_loss"].tolist() == pytest.approx([0.0, 1.0])


def test_missing_optional_policy_fields_are_reported_not_filtered() -> None:
    source = sample_dataframe().drop(
        columns=["receiving network coverage", "access point"],
    )
    prepared = prepare_candidates(
        filter_dataset(source, FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)
    constrained, report = apply_policy_constraints(
        losses,
        PolicyConstraints(
            min_coverage_score=1.0,
            allowed_access_points=("Bank branch",),
        ),
    )

    assert len(constrained) == 2
    assert report["removed_rows"] == 0
    assert "network_coverage_policy" in report["unsupported_constraints"]
    assert "access_point_policy" in report["unsupported_constraints"]


def test_policy_constraints_filter_cost_time_transparency_and_coverage() -> None:
    prepared = prepare_candidates(
        filter_dataset(sample_dataframe(), FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)
    constrained, report = apply_policy_constraints(
        losses,
        PolicyConstraints(
            max_total_cost_pct=2.0,
            max_settlement_days=1.0,
            require_transparency=True,
            min_coverage_score=0.9,
        ),
    )

    assert constrained["firm"].tolist() == ["FastCash"]
    assert report["eligible_rows"] == 1
    assert report["removed_rows"] == 1


def test_policy_constraints_filter_access_points() -> None:
    prepared = prepare_candidates(
        filter_dataset(sample_dataframe(), FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    losses = add_objective_losses(prepared)
    constrained, _report = apply_policy_constraints(
        losses,
        PolicyConstraints(allowed_access_points=("Bank branch",)),
    )

    assert constrained["firm"].tolist() == ["SlowBank"]


def test_batch_provider_concentration_limits_repeated_provider() -> None:
    prepared = prepare_candidates(
        filter_dataset(sample_dataframe(), FilterSpec(source_name="Kenya", destination_name="Uganda")),
        amount_tier="cc1",
    )
    scored = score_candidates(
        add_objective_losses(prepared),
        {"transaction_fee": 1.0, "risk": 0.0},
    )
    plans, report = build_batch_plans(
        scored,
        BatchSettings(transfer_count=2, max_provider_share=0.5, candidate_cap=2, plan_cap=8),
    )

    assert report["provider_limit"] == 1
    assert len(plans) == 1
    assert plans.iloc[0]["batch_provider_counts"] == {"FastCash": 1, "SlowBank": 1}


def test_settlement_days_maps_speed_labels_to_policy_values() -> None:
    assert settlement_days("Less than one hour") == pytest.approx(1.0 / 24.0)
    assert settlement_days("3-5 days") == 5.0


def test_parse_dates_accepts_both_source_formats() -> None:
    """The audited export is ISO; older extracts are day/month/year."""

    parsed = parse_dates(
        pd.Series(["24/Jan/2011", "2011-01-24", "2025-09-03", "not a date", None])
    )

    assert parsed.iloc[0] == pd.Timestamp("2011-01-24")
    assert parsed.iloc[1] == pd.Timestamp("2011-01-24")
    assert parsed.iloc[2] == pd.Timestamp("2025-09-03")
    assert pd.isna(parsed.iloc[3])
    assert pd.isna(parsed.iloc[4])


def test_hyphenated_pickup_column_is_aliased() -> None:
    """The audited export names the column ``PICK-UP METHOD``."""

    frame = pd.DataFrame({"PICK-UP METHOD": ["Cash"], "period": ["2025_3Q"]})
    normalized = ensure_supported_columns(frame, require_numeric=False)

    assert "pickup method" in normalized.columns
    assert normalized["pickup method"].tolist() == ["Cash"]


def test_mixed_date_formats_leave_no_unparsed_rows() -> None:
    """A column mixing both spellings must resolve completely."""

    parsed = parse_dates(pd.Series(["24/Jan/2011", "2025-09-03"] * 10))

    assert parsed.notna().all()


def test_qbraid_device_runtime_prefers_reported_execution_duration() -> None:
    """qBraid reports on-machine milliseconds; queue time must not leak into it."""

    from datetime import datetime, timedelta

    from qkash.quantum import _extract_qbraid_device_runtime

    class Stamps:
        createdAt = datetime(2026, 1, 1, 0, 0, 0)
        endedAt = datetime(2026, 1, 1, 0, 0, 30)   # 30 s wall clock incl. queue
        executionDuration = 250                     # 250 ms actually on the machine

    class Result:
        details = {"time_stamps": Stamps()}

    seconds, source = _extract_qbraid_device_runtime(Result())

    assert seconds == 0.25
    assert source == "qBraid executionDuration"


def test_qbraid_device_runtime_flags_queue_inclusive_fallback() -> None:
    """When the device reports nothing, the schema derives endedAt - createdAt."""

    from datetime import datetime

    from qkash.quantum import _extract_qbraid_device_runtime

    class Stamps:
        createdAt = datetime(2026, 1, 1, 0, 0, 0)
        endedAt = datetime(2026, 1, 1, 0, 0, 30)
        executionDuration = 30_000                  # exactly the wall-clock span

    class Result:
        details = {"time_stamps": Stamps()}

    seconds, source = _extract_qbraid_device_runtime(Result())

    assert seconds == 30.0
    assert "includes queue" in source


def test_qbraid_device_runtime_absent_is_reported_as_missing() -> None:
    class Result:
        details: dict[str, object] = {}

    from qkash.quantum import _extract_qbraid_device_runtime

    seconds, source = _extract_qbraid_device_runtime(Result())

    assert seconds is None
    assert source == "not reported by device"
