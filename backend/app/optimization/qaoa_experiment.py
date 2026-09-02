"""Reproducible multi-seed QAOA experiment orchestration and aggregation."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from backend.app.config import CLEANED_DATASET_PATH

from .classical_solver import solve_direct_argmin
from .constraint_mixer import validate_constraint_preserving_design
from .data_loader import load_cleaned_dataset
from .filters import build_eligible_alternatives
from .mathematical_solver import (
    MathematicalOptimizationResult,
    solve_constrained_with_scipy_milp,
)
from .models import Benchmark, ObjectiveWeights, SolverResult
from .qaoa_models import (
    DistributionSummary,
    QaoaAggregate,
    QaoaConfig,
    QaoaExperimentResult,
    QaoaMixer,
    QaoaRunResult,
)
from .qaoa_solver import run_qaoa
from .qubo import (
    DEFAULT_QUBO_PENALTY,
    build_qubo_model,
    is_penalty_sufficient,
    minimum_safe_penalty_threshold,
)
from .qubo_solver import solve_qubo_by_enumeration

REFERENCE_PERIOD = "2025_3Q"
REFERENCE_CORRIDOR = "KENTZA"
REFERENCE_SEEDS = (11, 29, 47, 71, 97)
REFERENCE_WEIGHTS = ObjectiveWeights(0.4, 0.3, 0.3)
RESULT_SCHEMA_VERSION = "2.0"
COMPARISON_SCHEMA_VERSION = "2.0"
SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS = ("1.0", RESULT_SCHEMA_VERSION)
COMPARISON_SLOT_EXPECTATIONS = {
    "standard_x_p1": (QaoaMixer.STANDARD_X, 1),
    "constraint_preserving_xy_p1": (QaoaMixer.CONSTRAINT_PRESERVING_XY, 1),
    "standard_x_p2": (QaoaMixer.STANDARD_X, 2),
    "constraint_preserving_xy_p2": (QaoaMixer.CONSTRAINT_PRESERVING_XY, 2),
}
RAW_DATASET_SHA256 = "7d3b394c0db9a8c4227cccd52bb73e09c026f3f1a74de6befdee8b6477ab1ce4"
PACKAGE_NAMES = (
    "qiskit",
    "qiskit-aer",
    "qiskit-algorithms",
    "qiskit-optimization",
    "scipy",
    "numpy",
)


def installed_package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _sha256_file() -> str:
    digest = sha256()
    with CLEANED_DATASET_PATH.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> DistributionSummary:
    if not values:
        return DistributionSummary(mean=None, standard_deviation=None)
    return DistributionSummary(
        mean=fmean(values),
        standard_deviation=pstdev(values),
    )


def aggregate_qaoa_runs(runs: tuple[QaoaRunResult, ...]) -> QaoaAggregate:
    if not runs:
        raise ValueError("At least one QAOA run is required for aggregation.")
    recovered = sum(run.exact_optimum_sampled for run in runs)
    modal_feasible = sum(run.raw_most_probable_state.feasible for run in runs)
    gaps = [
        run.absolute_optimality_gap
        for run in runs
        if run.absolute_optimality_gap is not None
    ]
    return QaoaAggregate(
        run_count=len(runs),
        successful_feasible_run_count=sum(run.success for run in runs),
        modal_state_feasibility_count=modal_feasible,
        modal_state_feasibility_rate=modal_feasible / len(runs),
        exact_optimum_recovery_count=recovered,
        exact_optimum_recovery_rate=recovered / len(runs),
        optimal_state_probability=_distribution(
            [run.optimal_state_probability for run in runs]
        ),
        feasible_probability=_distribution([run.feasible_probability for run in runs]),
        best_feasible_gap=_distribution(gaps),
        runtime_seconds=_distribution(
            [run.metrics.total_execution_time_seconds for run in runs]
        ),
        constraint_leakage_probability=_distribution(
            [run.constraint_leakage_probability for run in runs]
        ),
    )


def assert_mathematical_baseline_agreement(
    classical: SolverResult,
    mathematical: MathematicalOptimizationResult,
) -> None:
    """Accept any solver result belonging to the exact weighted-score tie set."""

    selected_id = (
        mathematical.selected_alternative.alternative_id
        if mathematical.selected_alternative is not None
        else None
    )
    if (
        not mathematical.success
        or not mathematical.feasible
        or selected_id not in classical.tie_information.tied_alternative_ids
        or abs(mathematical.weighted_score - classical.selected.weighted_score) > 1e-10
    ):
        raise AssertionError(
            "The mathematical-optimization baseline disagrees with the exact "
            "weighted-score optimum or its tie set."
        )


def run_qaoa_experiment(
    *,
    period: str = REFERENCE_PERIOD,
    corridor: str = REFERENCE_CORRIDOR,
    benchmark: Benchmark = Benchmark.CC2,
    weights: ObjectiveWeights = REFERENCE_WEIGHTS,
    penalty: float = DEFAULT_QUBO_PENALTY,
    reps: int = 1,
    shots: int = 2_048,
    max_iterations: int = 60,
    seeds: tuple[int, ...] = REFERENCE_SEEDS,
    mixer: QaoaMixer = QaoaMixer.STANDARD_X,
) -> QaoaExperimentResult:
    """Build the validated QUBO once, then run QAOA over configured seeds."""

    if not seeds:
        raise ValueError("At least one experiment seed is required.")
    alternatives = build_eligible_alternatives(
        load_cleaned_dataset(CLEANED_DATASET_PATH),
        period=period,
        corridor=corridor,
        benchmark=benchmark,
    )
    classical = solve_direct_argmin(alternatives, weights)
    model = build_qubo_model(alternatives, weights, penalty=penalty)
    minimum_penalty = minimum_safe_penalty_threshold(model.representation.scores)
    if not is_penalty_sufficient(model.representation.scores, penalty):
        raise ValueError(
            "QAOA experiment penalty must preserve the exactly-one feasible target: "
            f"received {penalty}, but this instance requires P > {minimum_penalty}."
        )
    mathematical = solve_constrained_with_scipy_milp(model)
    exact_qubo = solve_qubo_by_enumeration(model)
    assert_mathematical_baseline_agreement(classical, mathematical)
    runs = tuple(
        run_qaoa(
            model,
            QaoaConfig(
                reps=reps,
                shots=shots,
                max_iterations=max_iterations,
                algorithm_seed=seed,
                simulator_seed=seed,
                transpiler_seed=seed,
                mixer=mixer,
            ),
        )
        for seed in seeds
    )
    selected = exact_qubo.selected_alternative
    exact_reference: dict[str, Any] = {
        "classical_solver": classical.solver_name,
        "classical_runtime_seconds": classical.runtime_seconds,
        "bit_string_x0_to_xn": exact_qubo.bit_string,
        "qiskit_state_label_xn_to_x0": exact_qubo.bit_string[::-1],
        "qubo_energy": exact_qubo.energy,
        "weighted_score": classical.selected.weighted_score,
        "feasible": exact_qubo.feasible,
        "alternative_id": selected.alternative_id if selected else None,
        "provider": selected.firm if selected else None,
        "payment_instrument": selected.payment_instrument if selected else None,
        "pickup_method": selected.pickup_method if selected else None,
        "mathematical_optimization": {
            "solver": mathematical.solver_name,
            "runtime_seconds": mathematical.runtime_seconds,
            "success": mathematical.success,
            "status": mathematical.status,
            "message": mathematical.message,
            "feasible": mathematical.feasible,
            "bit_string_x0_to_xn": "".join(str(bit) for bit in mathematical.bit_values),
            "weighted_score": mathematical.weighted_score,
            "alternative_id": mathematical.selected_alternative.alternative_id,
            "mip_gap": mathematical.mip_gap,
            "mip_dual_bound": mathematical.mip_dual_bound,
            "mip_node_count": mathematical.mip_node_count,
        },
    }
    ranked_by_id = {
        item.alternative.alternative_id: item for item in model.ranked_alternatives
    }
    objective_inputs = [
        {
            "variable": variable,
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
        for variable, alternative, score in zip(
            model.representation.variable_order,
            model.alternatives,
            model.representation.scores,
            strict=True,
        )
    ]
    mixer_enum = QaoaMixer(mixer)
    constraint_validation = (
        asdict(validate_constraint_preserving_design(len(model.alternatives)))
        if mixer_enum is QaoaMixer.CONSTRAINT_PRESERVING_XY
        else None
    )
    configuration = {
        "period": period,
        "corridor": corridor,
        "benchmark": benchmark.value,
        "benchmark_amount": alternatives[0].benchmark_amount,
        "benchmark_currency": alternatives[0].benchmark_currency,
        "weights": asdict(weights),
        "penalty": penalty,
        "minimum_safe_penalty_condition": f"P > {minimum_penalty}",
        "reps": reps,
        "shots": shots,
        "optimizer": "COBYLA",
        "maximum_iterations": max_iterations,
        "seeds": list(seeds),
        "qubit_count": len(model.alternatives),
        "mixer": mixer_enum.value,
        "initial_state": (
            "uniform_one_hot_w_state"
            if mixer_enum is QaoaMixer.CONSTRAINT_PRESERVING_XY
            else "uniform_plus_state"
        ),
        "mixer_connectivity": (
            [list(edge) for edge in constraint_validation["connectivity"]]
            if constraint_validation is not None
            else []
        ),
        "constraint_preservation_validation": constraint_validation,
        "simulator": "qiskit-aer SamplerV2 (local, noiseless finite-shot)",
        "runtime_interpretation": (
            "Experimental implementation timing only; not evidence of quantum speedup."
        ),
        "provenance": {
            "raw_dataset_sha256_from_audit": RAW_DATASET_SHA256,
            "processed_dataset_path": str(
                CLEANED_DATASET_PATH.relative_to(CLEANED_DATASET_PATH.parents[2])
            ),
            "processed_dataset_sha256": _sha256_file(),
            "data_audit": "docs/data_audit.md",
            "normalization": {
                "method": "min-max within the eligible comparison set",
                "direction": "lower is better for fee, FX margin, and speed ordinal",
                "constant_feature_treatment": "all normalized values are zero",
                "bounds": {
                    "fee_percentage": {
                        "minimum": min(
                            item.fee_percentage for item in model.alternatives
                        ),
                        "maximum": max(
                            item.fee_percentage for item in model.alternatives
                        ),
                    },
                    "fx_margin": {
                        "minimum": min(item.fx_margin for item in model.alternatives),
                        "maximum": max(item.fx_margin for item in model.alternatives),
                    },
                    "speed_ordinal": {
                        "minimum": min(
                            item.speed_ordinal for item in model.alternatives
                        ),
                        "maximum": max(
                            item.speed_ordinal for item in model.alternatives
                        ),
                    },
                },
            },
            "objective_inputs_in_variable_order": objective_inputs,
        },
    }
    return QaoaExperimentResult(
        schema_version=RESULT_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        package_versions=installed_package_versions(),
        experiment_configuration=configuration,
        exact_reference=exact_reference,
        runs=runs,
        aggregate=aggregate_qaoa_runs(runs),
    )


def _comparison_distribution(values: list[float]) -> dict[str, float]:
    summary = _distribution(values)
    return asdict(summary)


def _comparison_entry(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    runs = payload["runs"]
    if not runs:
        raise ValueError(f"Comparison input {name!r} contains no runs.")
    leakage_values = [
        float(
            run.get("constraint_leakage_probability", 1.0 - run["feasible_probability"])
        )
        for run in runs
    ]
    modal_count = sum(run["raw_most_probable_state"]["feasible"] for run in runs)
    circuit_keys = ("original_depth", "decomposed_depth", "transpiled_depth")
    stored_mixer = payload["experiment_configuration"]["mixer"]
    mixer = (
        QaoaMixer.STANDARD_X.value
        if payload["schema_version"] == "1.0" and stored_mixer == "standard X mixer"
        else stored_mixer
    )
    return {
        "name": name,
        "mixer": mixer,
        "initial_state": payload["experiment_configuration"].get(
            "initial_state", "uniform_plus_state"
        ),
        "reps": payload["experiment_configuration"]["reps"],
        "seeds": payload["experiment_configuration"]["seeds"],
        "run_count": len(runs),
        "feasible_probability": _comparison_distribution(
            [float(run["feasible_probability"]) for run in runs]
        ),
        "constraint_leakage_probability": _comparison_distribution(leakage_values),
        "exact_optimum_recovery_count": sum(
            run["exact_optimum_sampled"] for run in runs
        ),
        "exact_optimum_recovery_rate": sum(run["exact_optimum_sampled"] for run in runs)
        / len(runs),
        "optimal_state_probability": _comparison_distribution(
            [float(run["optimal_state_probability"]) for run in runs]
        ),
        "best_feasible_gap": _comparison_distribution(
            [
                float(run["absolute_optimality_gap"])
                for run in runs
                if run["absolute_optimality_gap"] is not None
            ]
        ),
        "modal_state_feasibility_count": modal_count,
        "modal_state_feasibility_rate": modal_count / len(runs),
        "circuit_depth": {
            key: _comparison_distribution(
                [float(run["metrics"]["circuit"][key]) for run in runs]
            )
            for key in circuit_keys
        },
        "runtime_seconds": _comparison_distribution(
            [float(run["metrics"]["total_execution_time_seconds"]) for run in runs]
        ),
    }


def build_mixer_comparison(
    experiment_paths: dict[str, Path],
) -> dict[str, Any]:
    """Load compatible stored experiments and create a direct mixer comparison."""

    if not experiment_paths:
        raise ValueError("At least one experiment is required for comparison.")
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in experiment_paths.items()
    }
    for name, payload in payloads.items():
        schema_version = payload.get("schema_version")
        if schema_version not in SUPPORTED_EXPERIMENT_SCHEMA_VERSIONS:
            raise ValueError(
                f"Comparison input {name!r} uses unsupported experiment schema "
                f"{schema_version!r}."
            )
    first = next(iter(payloads.values()))
    problem_keys = (
        "period",
        "corridor",
        "benchmark",
        "benchmark_amount",
        "benchmark_currency",
        "weights",
        "penalty",
        "minimum_safe_penalty_condition",
        "qubit_count",
    )
    protocol_keys = ("shots", "optimizer", "maximum_iterations", "simulator")
    exact_keys = (
        "bit_string_x0_to_xn",
        "qiskit_state_label_xn_to_x0",
        "qubo_energy",
        "weighted_score",
        "feasible",
        "alternative_id",
        "provider",
        "payment_instrument",
        "pickup_method",
    )
    first_configuration = first["experiment_configuration"]
    expected_problem = {key: first_configuration[key] for key in problem_keys}
    expected_protocol = {key: first_configuration[key] for key in protocol_keys}
    expected_provenance = first_configuration["provenance"]
    expected_packages = first["package_versions"]
    expected_exact = {key: first["exact_reference"][key] for key in exact_keys}
    seeds_by_reps: dict[int, list[int]] = {}
    for name, payload in payloads.items():
        schema_version = payload["schema_version"]
        configuration = payload["experiment_configuration"]
        stored_mixer = configuration.get("mixer")
        valid_mixer = (
            stored_mixer == "standard X mixer"
            if schema_version == "1.0"
            else stored_mixer in {item.value for item in QaoaMixer}
        )
        if not valid_mixer:
            raise ValueError(
                f"Comparison input {name!r} has mixer {stored_mixer!r} invalid for "
                f"experiment schema {schema_version}."
            )
        canonical_mixer = (
            QaoaMixer.STANDARD_X
            if stored_mixer == "standard X mixer"
            else QaoaMixer(stored_mixer)
        )
        expected_slot = COMPARISON_SLOT_EXPECTATIONS.get(name)
        if expected_slot is not None and (
            canonical_mixer is not expected_slot[0]
            or int(configuration["reps"]) != expected_slot[1]
        ):
            raise ValueError(
                f"Comparison input {name!r} does not match its required mixer and depth."
            )
        observed_problem = {key: configuration[key] for key in problem_keys}
        observed_exact = {key: payload["exact_reference"][key] for key in exact_keys}
        if observed_problem != expected_problem or observed_exact != expected_exact:
            raise ValueError(
                f"Comparison input {name!r} uses a different problem instance."
            )
        if configuration["provenance"] != expected_provenance:
            raise ValueError(
                f"Comparison input {name!r} uses different data, normalization, "
                "or objective inputs."
            )
        observed_protocol = {key: configuration[key] for key in protocol_keys}
        if (
            observed_protocol != expected_protocol
            or payload["package_versions"] != expected_packages
        ):
            raise ValueError(
                f"Comparison input {name!r} uses a different execution protocol."
            )
        reps = int(configuration["reps"])
        seeds = [int(seed) for seed in configuration["seeds"]]
        run_seeds = [int(run["metrics"]["algorithm_seed"]) for run in payload["runs"]]
        if not seeds or len(set(seeds)) != len(seeds) or run_seeds != seeds:
            raise ValueError(
                f"Comparison input {name!r} has inconsistent configured and recorded seeds."
            )
        if reps in seeds_by_reps and seeds != seeds_by_reps[reps]:
            raise ValueError(
                f"Comparison input {name!r} uses a different seed set at p={reps}."
            )
        seeds_by_reps.setdefault(reps, seeds)
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_basis": {
            "problem": expected_problem,
            "execution_protocol": expected_protocol,
            "package_versions": expected_packages,
            "input_schema_versions": {
                name: payload["schema_version"] for name, payload in payloads.items()
            },
            "varying_fields_reported_per_experiment": [
                "mixer",
                "initial_state",
                "reps",
            ],
            "matched_seed_sets_by_reps": {
                str(reps): seeds for reps, seeds in sorted(seeds_by_reps.items())
            },
            "objective_inputs_in_variable_order": expected_provenance[
                "objective_inputs_in_variable_order"
            ],
        },
        "exact_weighted_score": expected_exact["weighted_score"],
        "processed_dataset_sha256": expected_provenance["processed_dataset_sha256"],
        "input_artifacts": {
            name: {
                "path": str(path),
                "sha256": _sha256_path(path),
            }
            for name, path in experiment_paths.items()
        },
        "interpretation": (
            "Formulation and feasibility comparison only; not evidence of quantum speedup "
            "or quantum advantage over classical computing."
        ),
        "experiments": [
            _comparison_entry(name, payload) for name, payload in payloads.items()
        ],
    }


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
