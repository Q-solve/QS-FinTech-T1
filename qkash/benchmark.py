"""Benchmark metrics and solver-output validation shared by all runs."""

from __future__ import annotations

# Imports.
from collections import Counter
import math
from typing import Iterable

import numpy as np
import pandas as pd


# Variable descriptions.
# EPSILON prevents divide-by-zero when a best objective is exactly zero.
EPSILON = 1e-12

# METRIC_RANKING_RULE orders solvers by quality first and execution cost second.
METRIC_RANKING_RULE = (
    ("relative_optimality_gap", True),
    ("objective_value", True),
    ("optimum_hit_probability", False),
    ("feasibility_rate", False),
    ("time_to_solution_s", True),
    ("end_to_end_runtime_s", True),
    ("stability", False),
)


def one_hot_index(bits: Iterable[int]) -> int | None:
    """Return the selected index if the bitstring is feasible."""

    vector = [int(bit) for bit in bits]
    if sum(vector) != 1:
        return None
    return vector.index(1)


def sample_from_bits(bits: Iterable[int], scores: Iterable[float]) -> dict[str, object]:
    """Convert a raw bitstring into the common benchmark sample shape."""

    bit_list = [int(bit) for bit in bits]
    score_vector = list(scores)
    index = one_hot_index(bit_list)
    objective = float(score_vector[index]) if index is not None else None
    return {
        "bits": bit_list,
        "index": index,
        "objective": objective,
        "feasible": index is not None,
    }


def deterministic_sample(index: int, scores: Iterable[float]) -> dict[str, object]:
    """Create a one-hot sample for deterministic algorithms."""

    score_vector = list(scores)
    bits = [0] * len(score_vector)
    bits[int(index)] = 1
    return sample_from_bits(bits, score_vector)


def summarize_samples(
    algorithm: str,
    samples: list[dict[str, object]],
    exact_objective: float,
    exact_indices: Iterable[int],
    runtime_s: float,
    confidence: float = 0.99,
) -> dict[str, object]:
    """Compute the core metrics requested for every optimizer."""

    if not samples:
        return {
            "algorithm": algorithm,
            "feasibility_rate": 0.0,
            "objective_value": np.nan,
            "relative_optimality_gap": np.nan,
            "optimum_hit_probability": 0.0,
            "end_to_end_runtime_s": runtime_s,
            "stability": 0.0,
            "time_to_solution_s": np.inf,
            "runs": 0,
        }

    exact_set = {int(index) for index in exact_indices}
    feasible = [sample for sample in samples if sample.get("feasible")]
    objectives = [
        float(sample["objective"])
        for sample in feasible
        if sample.get("objective") is not None
    ]
    best_objective = min(objectives) if objectives else np.nan
    gap = (
        (best_objective - exact_objective) / max(abs(exact_objective), EPSILON)
        if objectives
        else np.nan
    )
    hits = [
        sample
        for sample in samples
        if sample.get("feasible") and int(sample["index"]) in exact_set
    ]
    selected_indices = [int(sample["index"]) for sample in feasible]
    modal_share = 0.0
    if selected_indices:
        modal_share = Counter(selected_indices).most_common(1)[0][1] / len(selected_indices)

    hit_probability = len(hits) / len(samples)
    return {
        "algorithm": algorithm,
        "feasibility_rate": len(feasible) / len(samples),
        "objective_value": best_objective,
        "relative_optimality_gap": max(float(gap), 0.0) if not np.isnan(gap) else np.nan,
        "optimum_hit_probability": hit_probability,
        "end_to_end_runtime_s": float(runtime_s),
        "stability": float(modal_share),
        "time_to_solution_s": time_to_solution(runtime_s, len(samples), hit_probability, confidence),
        "runs": len(samples),
    }


def time_to_solution(
    runtime_s: float,
    runs: int,
    hit_probability: float,
    confidence: float = 0.99,
) -> float:
    """Estimate runtime needed to hit the optimum at the requested confidence."""

    if runs <= 0 or hit_probability <= 0:
        return np.inf
    runtime_per_run = runtime_s / runs
    if hit_probability >= 1.0:
        return runtime_per_run
    attempts = math.log(1.0 - confidence) / math.log(1.0 - hit_probability)
    return float(runtime_per_run * attempts)


def metrics_frame(metrics: list[dict[str, object]]) -> pd.DataFrame:
    """Format metrics into a predictable table for the UI."""

    columns = [
        "algorithm",
        "feasibility_rate",
        "objective_value",
        "relative_optimality_gap",
        "optimum_hit_probability",
        "end_to_end_runtime_s",
        "stability",
        "time_to_solution_s",
        "runs",
    ]
    return pd.DataFrame(metrics, columns=columns)


def rank_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank solvers by optimum quality, hit rate, feasibility, and runtime."""

    if metrics.empty:
        return metrics

    ranked = metrics.copy()
    for column, _ascending in METRIC_RANKING_RULE:
        if column not in ranked.columns:
            ranked[column] = np.nan

    return ranked.sort_values(
        by=[column for column, _ascending in METRIC_RANKING_RULE],
        ascending=[ascending for _column, ascending in METRIC_RANKING_RULE],
        na_position="last",
        kind="mergesort",
    )


def validate_solver_outputs(
    results: list[dict[str, object]],
    scores: Iterable[float],
    tolerance: float = 1e-9,
) -> list[dict[str, object]]:
    """Validate every solver result before metrics are trusted."""

    score_vector = list(float(score) for score in scores)
    return [
        validate_solver_result(result, score_vector, tolerance=tolerance)
        for result in results
    ]


def validate_solver_result(
    result: dict[str, object],
    scores: list[float],
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Validate one solver output against the candidate score vector."""

    algorithm = str(result.get("algorithm", "unknown"))
    status = str(result.get("status", "ok"))
    samples = result.get("samples", [])
    issues: list[str] = []

    if not isinstance(samples, list) or not samples:
        issues.append("solver returned no samples")

    best_index = result.get("best_index")
    if best_index is not None and not _valid_index(best_index, len(scores)):
        issues.append("best_index is outside candidate range")

    for sample_number, sample in enumerate(samples if isinstance(samples, list) else []):
        if not isinstance(sample, dict):
            issues.append(f"sample {sample_number} is not a dictionary")
            continue

        bits = sample.get("bits")
        if not isinstance(bits, list) or len(bits) != len(scores):
            issues.append(f"sample {sample_number} has invalid bit length")
            continue
        if any(bit not in {0, 1} for bit in bits):
            issues.append(f"sample {sample_number} has non-binary values")
            continue

        expected_index = one_hot_index(bits)
        reported_index = sample.get("index")
        reported_feasible = bool(sample.get("feasible", False))

        if reported_feasible != (expected_index is not None):
            issues.append(f"sample {sample_number} feasibility flag is inconsistent")

        if expected_index is None:
            if reported_index is not None:
                issues.append(f"sample {sample_number} reports an index for infeasible bits")
            continue

        if reported_index is None:
            issues.append(f"sample {sample_number} missing selected index")
            continue

        if int(reported_index) != expected_index:
            issues.append(f"sample {sample_number} selected index is inconsistent with bits")
            continue

        objective = sample.get("objective")
        if objective is None:
            issues.append(f"sample {sample_number} missing objective")
            continue

        if abs(float(objective) - scores[expected_index]) > tolerance:
            issues.append(f"sample {sample_number} objective does not match candidate score")

    return {
        "algorithm": algorithm,
        "status": status,
        "valid": not issues,
        "issues": "; ".join(issues) if issues else "ok",
    }


def _valid_index(value: object, size: int) -> bool:
    """Return whether a selected index points to an existing candidate."""

    try:
        index = int(value)
    except (TypeError, ValueError):
        return False
    return 0 <= index < size
