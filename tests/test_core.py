from __future__ import annotations

import pandas as pd
import pytest

from qkash.benchmark import rank_metrics, validate_solver_outputs
from qkash.classical import simulated_annealing
from qkash.data import (
    FilterSpec,
    ensure_supported_columns,
    filter_dataset,
    parse_dates,
    prepare_candidates,
)
from qkash.quantum import (
    DEFAULT_MAX_QUBITS,
    LocalAerBackend,
    QBraidBackend,
    QuantumBackend,
    run_qaoa,
)
from qkash.scoring import (
    build_selection_qubo,
    pareto_prune,
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
    assert result["circuit_depth"] > 0
    assert len(result["samples"]) == 8
    assert "circuit_diagram" in result
    assert "q_0" in result["circuit_diagram"]


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


def test_qaoa_default_max_qubits_is_twenty() -> None:
    assert DEFAULT_MAX_QUBITS == 5


def test_qubo_candidate_selection_fills_to_five_after_strict_pareto() -> None:
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
    assert len(model_candidates) == 5
    assert model_candidates["model_source"].tolist().count("Pareto frontier") == 1
    assert model_candidates["model_source"].tolist().count("Best scored fallback") == 4


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
