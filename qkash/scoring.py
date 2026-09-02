"""Weighted scoring, Pareto pruning, and QUBO construction."""

from __future__ import annotations

# Imports.
from dataclasses import dataclass
import itertools
import re
from typing import Iterable

import numpy as np
import pandas as pd


# Variable descriptions.
# DEFAULT_WEIGHTS controls the three user-adjustable optimization priorities.
DEFAULT_WEIGHTS: dict[str, float] = {
    "transaction_fee": 0.34,
    "time": 0.33,
    "fx_spread": 0.33,
}

# LOSS_COLUMNS are the lower-is-better objective columns used for Pareto pruning.
LOSS_COLUMNS = (
    "transaction_fee_loss",
    "time_loss",
    "fx_spread_loss",
)

# CANDIDATE_POOL_INDEX_COLUMN keeps scored rows uniquely identifiable after pruning.
CANDIDATE_POOL_INDEX_COLUMN = "_candidate_pool_index"

# MODEL_SOURCE_COLUMN explains why a row entered the final fixed-size QUBO.
MODEL_SOURCE_COLUMN = "model_source"

# KEYWORDS map plain-language query terms to the three objective weights.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "transaction_fee": (
        "transaction fee",
        "fee",
        "fees",
        "commission",
        "charge",
        "charges",
        "cheap",
        "cheapest",
        "affordable",
    ),
    "time": ("time", "fast", "faster", "urgent", "instant", "same day", "quick", "speed"),
    "fx_spread": (
        "fx spread",
        "fx",
        "foreign exchange",
        "exchange",
        "spread",
        "margin",
        "rate",
    ),
}


@dataclass(frozen=True)
class QuboModel:
    """Upper-triangular QUBO matrix for the one-provider selection model."""

    matrix: np.ndarray
    offset: float
    penalty: float
    scores: np.ndarray
    labels: tuple[str, ...] = ()

    def energy(self, bits: Iterable[int]) -> float:
        vector = np.asarray(list(bits), dtype=float)
        upper_energy = float(np.sum(np.triu(self.matrix) * np.outer(vector, vector)))
        return self.offset + upper_energy

    @property
    def size(self) -> int:
        return int(self.scores.size)


def parse_weight_query(query: str) -> dict[str, float]:
    """Turn a plain-language priority query into normalized weights.

    The parser supports keyword intent such as "low fee, fast, low FX spread"
    and explicit overrides such as ``fee=0.5 time=0.3 fx=0.2``.
    """

    text = (query or "").lower()
    if not text.strip():
        return dict(DEFAULT_WEIGHTS)

    bumps = {name: 0.0 for name in DEFAULT_WEIGHTS}
    for name, words in KEYWORDS.items():
        bumps[name] += sum(1.0 for word in words if word in text)

    if any(value > 0 for value in bumps.values()):
        weights = {name: 0.05 + bumps[name] for name in DEFAULT_WEIGHTS}
    else:
        weights = dict(DEFAULT_WEIGHTS)

    aliases = {
        "transaction_fee": ("transaction_fee", "transaction fee", "fee", "fees"),
        "time": ("time", "speed"),
        "fx_spread": ("fx_spread", "fx spread", "fx", "spread"),
    }
    for name, names in aliases.items():
        alias_pattern = "|".join(re.escape(alias) for alias in names)
        match = re.search(rf"\b(?:{alias_pattern})\s*[:=]\s*(\d+(?:\.\d+)?)", text)
        if match:
            weights[name] = float(match.group(1))

    return normalize_weights(weights)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Keep known weight names, clamp negatives, and normalize to one."""

    cleaned = {name: max(float(weights.get(name, 0.0)), 0.0) for name in DEFAULT_WEIGHTS}
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def score_candidates(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Add normalized loss columns and the final weighted score.

    Lower weighted scores are better.  Time is represented by the mapped speed
    score, then inverted so faster services have lower loss.
    """

    if candidates.empty:
        return candidates.copy()

    normalized = normalize_weights(weights)
    scored = candidates.copy()
    scored["transaction_fee_loss"] = _minmax_loss(scored["fee_lcu"])
    scored["time_loss"] = 1.0 - scored["speed_score"].clip(0, 1)
    scored["fx_spread_loss"] = _minmax_loss(scored["fx_margin"])

    scored["weighted_score"] = (
        normalized["transaction_fee"] * scored["transaction_fee_loss"]
        + normalized["time"] * scored["time_loss"]
        + normalized["fx_spread"] * scored["fx_spread_loss"]
    )
    scored["weighted_score"] = scored["weighted_score"].astype(float)
    return scored.sort_values("weighted_score", kind="mergesort").reset_index(drop=True)


def pareto_prune(
    candidates: pd.DataFrame,
    columns: tuple[str, ...] = LOSS_COLUMNS,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Remove candidates dominated on all objective-loss dimensions."""

    if candidates.empty:
        return candidates.copy()

    points = candidates.loc[:, columns].astype(float).to_numpy()
    dominated = np.zeros(len(points), dtype=bool)

    for i, point in enumerate(points):
        if dominated[i]:
            continue
        no_worse = np.all(points <= point + tolerance, axis=1)
        strictly_better = np.any(points < point - tolerance, axis=1)
        dominated[i] = bool(np.any(no_worse & strictly_better))

    return (
        candidates.loc[~dominated]
        .sort_values("weighted_score", kind="mergesort")
        .reset_index(drop=True)
    )


def select_qubo_candidates(
    scored: pd.DataFrame,
    target_size: int,
    columns: tuple[str, ...] = LOSS_COLUMNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixed-size QUBO candidate set from scored and Pareto rows.

    Strict Pareto pruning can sometimes leave fewer rows than the requested
    qubit count.  In that case, the missing slots are filled with the next-best
    scored candidates from the already-filtered dataset, so every qubit still
    represents a real candidate row.
    """

    requested = int(target_size)
    if requested <= 0:
        raise ValueError("target_size must be positive")

    if scored.empty:
        empty = scored.copy()
        empty[MODEL_SOURCE_COLUMN] = pd.Series(dtype=str)
        return empty, empty.copy()

    pool = scored.sort_values("weighted_score", kind="mergesort").reset_index(drop=True).copy()
    pool[CANDIDATE_POOL_INDEX_COLUMN] = np.arange(len(pool), dtype=int)

    pruned = pareto_prune(pool, columns=columns)
    frontier = pruned.head(requested).copy()
    frontier[MODEL_SOURCE_COLUMN] = "Pareto frontier"

    remaining = requested - len(frontier)
    if remaining > 0:
        used_pool_indices = set(frontier[CANDIDATE_POOL_INDEX_COLUMN].astype(int).tolist())
        fallback = pool.loc[
            ~pool[CANDIDATE_POOL_INDEX_COLUMN].isin(used_pool_indices)
        ].head(remaining).copy()
        fallback[MODEL_SOURCE_COLUMN] = "Best scored fallback"
        selected = pd.concat([frontier, fallback], ignore_index=True)
    else:
        selected = frontier

    selected = (
        selected.head(requested)
        .sort_values("weighted_score", kind="mergesort")
        .reset_index(drop=True)
    )
    return selected, pruned


def build_selection_qubo(
    scores: Iterable[float],
    penalty: float | None = None,
    labels: Iterable[str] | None = None,
) -> QuboModel:
    """Build QUBO for ``min c'x`` subject to selecting exactly one candidate."""

    score_vector = np.asarray(list(scores), dtype=float)
    if score_vector.size == 0:
        raise ValueError("Cannot build a QUBO with zero candidates")
    if np.any(score_vector < -1e-12):
        raise ValueError("QKash expects non-negative weighted scores")

    chosen_penalty = float(penalty) if penalty is not None else max(1.0, 2.0 * score_vector.max())
    if chosen_penalty <= 0:
        raise ValueError("penalty must be positive")

    size = score_vector.size
    matrix = np.zeros((size, size), dtype=float)
    for i, score in enumerate(score_vector):
        matrix[i, i] = score - chosen_penalty
    for i in range(size):
        for j in range(i + 1, size):
            matrix[i, j] = 2.0 * chosen_penalty

    label_tuple = tuple(labels) if labels is not None else tuple(str(i) for i in range(size))
    return QuboModel(
        matrix=matrix,
        offset=chosen_penalty,
        penalty=chosen_penalty,
        scores=score_vector,
        labels=label_tuple,
    )


def solve_original_exact(scores: Iterable[float]) -> dict[str, object]:
    """Solve the constrained one-hot mathematical baseline exactly."""

    score_vector = np.asarray(list(scores), dtype=float)
    if score_vector.size == 0:
        raise ValueError("No scores supplied")
    best_value = float(np.min(score_vector))
    best_indices = np.flatnonzero(np.isclose(score_vector, best_value)).astype(int).tolist()
    return {"objective": best_value, "indices": best_indices, "index": best_indices[0]}


def verify_qubo_equivalence(qubo: QuboModel, exhaustive_limit: int = 20) -> dict[str, object]:
    """Verify that the QUBO and constrained model have the same optimum.

    For small instances this enumerates all bitstrings.  For larger instances it
    uses the one-hot penalty structure: since all weighted scores are
    non-negative, every infeasible state has energy at least ``penalty`` and
    every feasible one-hot state has energy equal to its original score.
    """

    exact = solve_original_exact(qubo.scores)
    min_score = float(exact["objective"])

    if qubo.size <= exhaustive_limit:
        best_energy = np.inf
        best_bits: list[tuple[int, ...]] = []
        for bits in itertools.product((0, 1), repeat=qubo.size):
            energy = qubo.energy(bits)
            if energy < best_energy - 1e-12:
                best_energy = energy
                best_bits = [bits]
            elif np.isclose(energy, best_energy):
                best_bits.append(bits)

        qubo_indices = [
            bits.index(1) for bits in best_bits if sum(bits) == 1 and 1 in bits
        ]
        is_equivalent = (
            np.isclose(best_energy, min_score)
            and set(qubo_indices) == set(exact["indices"])
        )
        return {
            "equivalent": bool(is_equivalent),
            "method": "exhaustive",
            "original_objective": min_score,
            "qubo_objective": float(best_energy),
            "original_indices": exact["indices"],
            "qubo_indices": sorted(qubo_indices),
            "message": "QUBO optimum matches the constrained one-hot model."
            if is_equivalent
            else "QUBO optimum does not match the constrained one-hot model.",
        }

    is_equivalent = qubo.penalty > min_score
    return {
        "equivalent": bool(is_equivalent),
        "method": "analytic",
        "original_objective": min_score,
        "qubo_objective": min_score if is_equivalent else None,
        "original_indices": exact["indices"],
        "qubo_indices": exact["indices"] if is_equivalent else [],
        "message": "QUBO equivalence verified analytically from the penalty structure."
        if is_equivalent
        else "Penalty is too small to exclude infeasible QUBO states.",
    }


def _minmax_loss(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    filled = values.fillna(values.median())
    minimum = float(filled.min())
    maximum = float(filled.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(filled)), index=series.index)
    return (filled - minimum) / (maximum - minimum)
