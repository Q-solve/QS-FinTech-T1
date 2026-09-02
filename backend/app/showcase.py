"""Read-only assembly of the validated Remit-Q showcase payload."""

from __future__ import annotations

import json
from dataclasses import asdict
from functools import lru_cache
from hashlib import sha256

from .config import CLEANED_DATASET_PATH, PROJECT_ROOT, QAOA_COMPARISON_PATH
from .optimization.classical_solver import solve_direct_argmin
from .optimization.data_loader import load_cleaned_dataset
from .optimization.filters import build_eligible_alternatives
from .optimization.models import Benchmark, ObjectiveWeights
from .optimization.qubo import DEFAULT_QUBO_PENALTY, build_qubo_model
from .schemas import (
    DistributionResponse,
    ObjectiveWeightsResponse,
    ShowcaseAlternativeResponse,
    ShowcaseExperimentResponse,
    ShowcaseProtocolResponse,
    ShowcaseResponse,
    ShowcaseScenarioResponse,
)

SHOWCASE_WEIGHTS = ObjectiveWeights(0.4, 0.3, 0.3)
EXPERIMENT_LABELS = {
    "standard_x_p1": "Standard X · p=1",
    "constraint_preserving_xy_p1": "XY + W · p=1",
    "standard_x_p2": "Standard X · p=2",
    "constraint_preserving_xy_p2": "XY + W · p=2",
}


def _validate_comparison(
    comparison: dict[str, object], objective_inputs: list[dict[str, object]]
) -> None:
    """Reject stored evidence that does not describe the live showcase instance."""

    current_hash = sha256(CLEANED_DATASET_PATH.read_bytes()).hexdigest()
    if comparison["processed_dataset_sha256"] != current_hash:
        raise ValueError(
            "QAOA results were produced from a different processed dataset."
        )
    for name, artifact in comparison["input_artifacts"].items():
        artifact_path = (PROJECT_ROOT / artifact["path"]).resolve()
        if not artifact_path.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"QAOA input artifact {name!r} is outside the project.")
        observed_hash = sha256(artifact_path.read_bytes()).hexdigest()
        if observed_hash != artifact["sha256"]:
            raise ValueError(f"QAOA input artifact {name!r} has changed.")

    problem = comparison["comparison_basis"]["problem"]
    expected = {
        "period": "2025_3Q",
        "corridor": "KENTZA",
        "benchmark": Benchmark.CC2.value,
        "benchmark_amount": 45_000.0,
        "benchmark_currency": "KES",
        "qubit_count": 14,
    }
    for field, value in expected.items():
        if problem[field] != value:
            raise ValueError(f"QAOA comparison has an unexpected {field} value.")
    artifact_weights = problem["weights"]
    expected_weights = {
        "fee_weight": SHOWCASE_WEIGHTS.fee_weight,
        "fx_weight": SHOWCASE_WEIGHTS.fx_weight,
        "speed_weight": SHOWCASE_WEIGHTS.speed_weight,
    }
    if artifact_weights != expected_weights:
        raise ValueError("QAOA comparison weights do not match the showcase objective.")
    if (
        comparison["comparison_basis"]["objective_inputs_in_variable_order"]
        != objective_inputs
    ):
        raise ValueError(
            "QAOA results use different objective inputs or variable order."
        )


def _distribution(payload: dict[str, float | None]) -> DistributionResponse:
    return DistributionResponse(
        mean=payload["mean"],
        standard_deviation=payload["standard_deviation"],
    )


@lru_cache(maxsize=1)
def build_showcase() -> ShowcaseResponse:
    """Build the fixed, source-backed demonstration without running QAOA on request."""

    alternatives = build_eligible_alternatives(
        load_cleaned_dataset(CLEANED_DATASET_PATH),
        period="2025_3Q",
        corridor="KENTZA",
        benchmark=Benchmark.CC2,
    )
    classical = solve_direct_argmin(alternatives, SHOWCASE_WEIGHTS)
    model = build_qubo_model(
        alternatives, SHOWCASE_WEIGHTS, penalty=DEFAULT_QUBO_PENALTY
    )
    ranked_by_id = {
        item.alternative.alternative_id: item for item in classical.ranked_alternatives
    }
    objective_inputs = [
        {
            "variable": f"x_{index}",
            "alternative_id": alternative.alternative_id,
            "provider": alternative.firm,
            "payment_instrument": alternative.payment_instrument,
            "pickup_method": alternative.pickup_method,
            "fee_percentage": alternative.fee_percentage,
            "fx_margin": alternative.fx_margin,
            "speed_ordinal": alternative.speed_ordinal,
            "normalized": asdict(ranked_by_id[alternative.alternative_id].normalized),
            "weighted_score": score,
        }
        for index, (alternative, score) in enumerate(
            zip(model.alternatives, model.representation.scores, strict=True)
        )
    ]
    selected_id = classical.selected.alternative.alternative_id
    ranked = [
        ShowcaseAlternativeResponse(
            rank=item.rank,
            alternative_id=item.alternative.alternative_id,
            provider=item.alternative.firm,
            provider_type=item.alternative.firm_type,
            payment_instrument=item.alternative.payment_instrument,
            pickup_method=item.alternative.pickup_method,
            speed=item.alternative.speed_label,
            fee_percentage=float(item.alternative.fee_percentage),
            fx_margin=float(item.alternative.fx_margin),
            total_cost_percentage=float(item.alternative.total_cost_percentage),
            weighted_score=float(item.weighted_score),
            selected=item.alternative.alternative_id == selected_id,
        )
        for item in classical.ranked_alternatives
    ]
    comparison = json.loads(QAOA_COMPARISON_PATH.read_text(encoding="utf-8"))
    _validate_comparison(comparison, objective_inputs)
    basis = comparison["comparison_basis"]
    execution = basis["execution_protocol"]
    experiments = [
        ShowcaseExperimentResponse(
            key=item["name"],
            label=EXPERIMENT_LABELS[item["name"]],
            mixer=item["mixer"],
            depth=int(item["reps"]),
            run_count=int(item["run_count"]),
            feasible_probability=_distribution(item["feasible_probability"]),
            leakage_probability=_distribution(item["constraint_leakage_probability"]),
            optimal_probability=_distribution(item["optimal_state_probability"]),
            exact_recovery_count=int(item["exact_optimum_recovery_count"]),
            exact_recovery_rate=float(item["exact_optimum_recovery_rate"]),
            modal_feasibility_rate=float(item["modal_state_feasibility_rate"]),
            best_feasible_gap=_distribution(item["best_feasible_gap"]),
            transpiled_depth=_distribution(item["circuit_depth"]["transpiled_depth"]),
            runtime_seconds=_distribution(item["runtime_seconds"]),
        )
        for item in comparison["experiments"]
    ]
    first = alternatives[0]
    return ShowcaseResponse(
        title="Quantum-assisted remittance service selection",
        scenario=ShowcaseScenarioResponse(
            period=first.period,
            corridor=first.corridor,
            source="Kenya",
            destination="Tanzania",
            story_destination="Zanzibar (represented by Tanzania in the dataset)",
            benchmark=first.benchmark.value,
            benchmark_amount=float(first.benchmark_amount),
            benchmark_currency=first.benchmark_currency,
            alternative_count=len(alternatives),
        ),
        weights=ObjectiveWeightsResponse(fee=0.4, fx_margin=0.3, speed=0.3),
        recommendation=next(item for item in ranked if item.selected),
        alternatives=ranked,
        experiments=experiments,
        protocol=ShowcaseProtocolResponse(
            shots=int(execution["shots"]),
            optimizer=execution["optimizer"],
            maximum_iterations=int(execution["maximum_iterations"]),
            simulator=execution["simulator"],
            matched_seeds_by_depth=basis["matched_seed_sets_by_reps"],
        ),
        workflow=[
            "Filter the audited dataset to 2025_3Q, Kenya→Tanzania, and the CC2 KES 45,000 benchmark.",
            "Create 14 exact-unique provider/service alternatives and normalize fee, FX margin, and speed within that set.",
            "Apply weights 0.4, 0.3, and 0.3; direct argmin, exhaustive search, and mathematical optimization establish the exact baseline.",
            "Encode the identical score and exactly-one constraint as the validated QUBO/Ising cost Hamiltonian.",
            "Compare standard X-QAOA with W-state plus ring-XY QAOA over matched seeds, then report feasibility, leakage, recovery, depth, and runtime.",
        ],
        limitations=[
            "This is a direct one-of-14 choice and remains classically trivial; the results do not establish quantum advantage.",
            "KES 45,000 is the source-supported CC2 benchmark. The interface does not present a fabricated KES 100,000 quote.",
            "Results are local, noiseless finite-shot simulations, not quantum-hardware measurements.",
            "The constraint-preserving circuit is deeper; simulator runtime is not evidence of speedup.",
        ],
        provenance={
            "dataset": "World Bank remittance price workbook, audited derived CSV",
            "period": "2025_3Q",
            "processed_sha256": comparison["processed_dataset_sha256"],
            "comparison_schema": comparison["schema_version"],
            "results": "experiments/results/qaoa_mixer_comparison.json",
        },
    )
