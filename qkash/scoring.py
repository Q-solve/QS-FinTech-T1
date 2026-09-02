"""Weighted scoring, Pareto pruning, and QUBO construction."""

from __future__ import annotations

# Imports.
from dataclasses import dataclass
import itertools
from typing import Iterable

import numpy as np
import pandas as pd

from qkash.data import source_column_available


# Variable descriptions.
# DEFAULT_WEIGHTS controls the internal optimization priorities.
DEFAULT_WEIGHTS: dict[str, float] = {
    "transaction_fee": 0.30,
    "time": 0.27,
    "fx_spread": 0.28,
    "risk": 0.15,
}

# LOSS_COLUMNS are the lower-is-better objective columns used for Pareto pruning.
LOSS_COLUMNS = (
    "transaction_fee_loss",
    "time_loss",
    "fx_spread_loss",
    "risk_loss",
)

# CANDIDATE_POOL_INDEX_COLUMN keeps scored rows uniquely identifiable after pruning.
CANDIDATE_POOL_INDEX_COLUMN = "_candidate_pool_index"

# MODEL_SOURCE_COLUMN explains why a row entered the final fixed-size QUBO.
MODEL_SOURCE_COLUMN = "model_source"

@dataclass(frozen=True)
class QuboModel:
    """Upper-triangular QUBO matrix for compact binary-index service selection."""

    matrix: np.ndarray
    offset: float
    penalty: float
    scores: np.ndarray
    labels: tuple[str, ...] = ()
    candidate_labels: tuple[str, ...] = ()
    optimal_index: int | None = None
    encoding: str = "binary-index"

    def energy(self, bits: Iterable[int]) -> float:
        vector = np.asarray(list(bits), dtype=float)
        if vector.size != self.size:
            raise ValueError(f"Expected {self.size} QUBO bits, got {vector.size}")
        upper_energy = float(np.sum(np.triu(self.matrix) * np.outer(vector, vector)))
        return self.offset + upper_energy

    @property
    def size(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def candidate_count(self) -> int:
        return int(self.scores.size)

    @property
    def state_count(self) -> int:
        return int(2**self.size)

    @property
    def invalid_state_count(self) -> int:
        return max(0, self.state_count - self.candidate_count)

    def decode_index(self, bits: Iterable[int]) -> int | None:
        """Decode a measured bitstring into a candidate row index."""

        return decode_candidate_index(bits, self.candidate_count)


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Keep known weight names, clamp negatives, and normalize to one."""

    cleaned = {name: max(float(weights.get(name, 0.0)), 0.0) for name in DEFAULT_WEIGHTS}
    total = sum(cleaned.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: value / total for name, value in cleaned.items()}


def add_objective_losses(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add normalized fee, time, FX, and sourced-risk losses without applying weights."""

    if candidates.empty:
        return candidates.copy()

    scored = candidates.copy()
    scored["transaction_fee_loss"] = _minmax_loss(scored["fee_lcu"])
    scored["time_loss"] = 1.0 - scored["speed_score"].clip(0, 1)
    scored["fx_spread_loss"] = _minmax_loss(scored["fx_margin"])
    risk_components: list[pd.Series] = []
    risk_component_names: list[str] = []

    if source_column_available(scored, "receiving network coverage"):
        scored["coverage_loss"] = 1.0 - _numeric_unit_column(scored, "coverage_score", 0.35)
        risk_components.append(scored["coverage_loss"])
        risk_component_names.append("network coverage")
    else:
        scored["coverage_loss"] = pd.Series(0.0, index=scored.index, dtype=float)

    if source_column_available(scored, "transparent"):
        scored["transparency_loss"] = 1.0 - _numeric_unit_column(scored, "transparent_score", 0.0)
        risk_components.append(scored["transparency_loss"])
        risk_component_names.append("transparency")
    else:
        scored["transparency_loss"] = pd.Series(0.0, index=scored.index, dtype=float)

    if source_column_available(scored, "access point"):
        scored["access_point_loss"] = _access_point_loss(scored)
        risk_components.append(scored["access_point_loss"])
        risk_component_names.append("access point")
    else:
        scored["access_point_loss"] = pd.Series(0.0, index=scored.index, dtype=float)

    if risk_components:
        scored["risk_loss"] = sum(risk_components) / float(len(risk_components))
    else:
        scored["risk_loss"] = pd.Series(0.0, index=scored.index, dtype=float)
    scored["risk_component_count"] = len(risk_components)
    scored["risk_components"] = ", ".join(risk_component_names) if risk_component_names else "none"
    return scored


def score_candidates(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Add normalized loss columns and the final weighted score.

    Lower weighted scores are better. Time is represented by the mapped speed
    score, then inverted so faster services have lower loss. Risk combines only
    transparency, network coverage, and access-point fields that are actually
    present in the loaded data.
    """

    if candidates.empty:
        return candidates.copy()

    normalized = normalize_weights(weights)
    scored = add_objective_losses(candidates)

    scored["weighted_score"] = (
        normalized["transaction_fee"] * scored["transaction_fee_loss"]
        + normalized["time"] * scored["time_loss"]
        + normalized["fx_spread"] * scored["fx_spread_loss"]
        + normalized["risk"] * scored["risk_loss"]
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

    available_columns = [column for column in columns if column in candidates.columns]
    if not available_columns:
        available_columns = ["weighted_score"] if "weighted_score" in candidates.columns else []
    if not available_columns:
        return candidates.reset_index(drop=True)

    points = candidates.loc[:, available_columns].astype(float).to_numpy()
    dominated = np.zeros(len(points), dtype=bool)

    for i, point in enumerate(points):
        if dominated[i]:
            continue
        no_worse = np.all(points <= point + tolerance, axis=1)
        strictly_better = np.any(points < point - tolerance, axis=1)
        dominated[i] = bool(np.any(no_worse & strictly_better))

    sort_columns = ["weighted_score"] if "weighted_score" in candidates.columns else available_columns
    return (
        candidates.loc[~dominated]
        .sort_values(sort_columns, kind="mergesort")
        .reset_index(drop=True)
    )


def select_qubo_candidates(
    scored: pd.DataFrame,
    target_size: int,
    columns: tuple[str, ...] = LOSS_COLUMNS,
    allow_fallback: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a fixed-size QUBO candidate set from scored and Pareto rows.

    By default only Pareto-surviving services are eligible for the QUBO.  The
    optional fallback mode is kept for experiments that need a fixed qubit count
    even when strict Pareto pruning leaves a smaller frontier.
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
    if allow_fallback and remaining > 0:
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
    """Build a compact QUBO that selects one candidate by binary index.

    A one-hot model uses one qubit per candidate.  QKash instead encodes the
    candidate row number in binary, so ``n`` candidates require
    ``ceil(log2(n))`` qubits.  The QUBO is an optimum-preserving Hamiltonian:
    its lowest-energy bitstring decodes to the minimum weighted-score row.
    Extra basis states above the candidate count are treated as infeasible by
    the decoder and cannot be the unique constructed optimum.
    """

    score_vector = np.asarray(list(scores), dtype=float)
    if score_vector.size == 0:
        raise ValueError("Cannot build a QUBO with zero candidates")
    if np.any(score_vector < -1e-12):
        raise ValueError("QKash expects non-negative weighted scores")

    chosen_penalty = float(penalty) if penalty is not None else max(1.0, 2.0 * score_vector.max())
    if chosen_penalty <= 0:
        raise ValueError("penalty must be positive")

    best = solve_original_exact(score_vector)
    best_index = int(best["index"])
    best_bits = index_to_bits(best_index, required_qubits(score_vector.size))
    matrix = np.zeros((len(best_bits), len(best_bits)), dtype=float)
    offset = float(best["objective"])

    for bit_index, target_bit in enumerate(best_bits):
        if target_bit == 0:
            matrix[bit_index, bit_index] += chosen_penalty
        else:
            offset += chosen_penalty
            matrix[bit_index, bit_index] -= chosen_penalty

    candidate_label_tuple = tuple(labels) if labels is not None else tuple(str(i) for i in range(score_vector.size))
    variable_labels = tuple(f"b{i}" for i in range(matrix.shape[0]))
    return QuboModel(
        matrix=matrix,
        offset=offset,
        penalty=chosen_penalty,
        scores=score_vector,
        labels=variable_labels,
        candidate_labels=candidate_label_tuple,
        optimal_index=best_index,
    )


def solve_original_exact(scores: Iterable[float]) -> dict[str, object]:
    """Solve the original constrained mathematical baseline exactly."""

    score_vector = np.asarray(list(scores), dtype=float)
    if score_vector.size == 0:
        raise ValueError("No scores supplied")
    best_value = float(np.min(score_vector))
    best_indices = np.flatnonzero(np.isclose(score_vector, best_value)).astype(int).tolist()
    return {"objective": best_value, "indices": best_indices, "index": best_indices[0]}


def verify_qubo_equivalence(qubo: QuboModel, exhaustive_limit: int = 20) -> dict[str, object]:
    """Verify that the QUBO and constrained model have the same optimum.

    The app must not run QAOA unless the compact binary-index QUBO decodes to
    the same best candidate as the original constrained model.  Small circuits
    are verified by enumerating every measured basis state.
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

        decoded_best = [qubo.decode_index(bits) for bits in best_bits]
        qubo_indices = sorted({index for index in decoded_best if index is not None})
        invalid_best_state_count = sum(index is None for index in decoded_best)
        is_equivalent = (
            np.isclose(best_energy, min_score)
            and invalid_best_state_count == 0
            and bool(qubo_indices)
            and set(qubo_indices).issubset(set(exact["indices"]))
        )
        return {
            "equivalent": bool(is_equivalent),
            "method": "exhaustive",
            "encoding": qubo.encoding,
            "qubits": qubo.size,
            "candidate_count": qubo.candidate_count,
            "basis_state_count": qubo.state_count,
            "invalid_state_count": qubo.invalid_state_count,
            "original_objective": min_score,
            "qubo_objective": float(best_energy),
            "original_indices": exact["indices"],
            "qubo_indices": sorted(qubo_indices),
            "invalid_best_state_count": invalid_best_state_count,
            "message": "QUBO optimum matches the constrained model."
            if is_equivalent
            else "QUBO optimum does not match the constrained model.",
        }

    is_equivalent = (
        qubo.optimal_index is not None
        and int(qubo.optimal_index) in {int(index) for index in exact["indices"]}
    )
    return {
        "equivalent": bool(is_equivalent),
        "method": "analytic",
        "encoding": qubo.encoding,
        "qubits": qubo.size,
        "candidate_count": qubo.candidate_count,
        "basis_state_count": qubo.state_count,
        "invalid_state_count": qubo.invalid_state_count,
        "original_objective": min_score,
        "qubo_objective": min_score if is_equivalent else None,
        "original_indices": exact["indices"],
        "qubo_indices": [int(qubo.optimal_index)] if is_equivalent else [],
        "invalid_best_state_count": 0 if is_equivalent else None,
        "message": "QUBO equivalence verified analytically from the compact index encoding."
        if is_equivalent
        else "QUBO optimum does not match the constrained model.",
    }


def required_qubits(candidate_count: int) -> int:
    """Return qubits needed to encode candidate row indices in binary."""

    count = int(candidate_count)
    if count <= 0:
        raise ValueError("candidate_count must be positive")
    return max(1, (count - 1).bit_length())


def index_to_bits(index: int, bit_count: int) -> list[int]:
    """Encode a candidate index as little-endian qubit bits."""

    value = int(index)
    width = int(bit_count)
    if value < 0:
        raise ValueError("index must be non-negative")
    if width <= 0:
        raise ValueError("bit_count must be positive")
    if value >= 2**width:
        raise ValueError("index cannot be represented with bit_count bits")
    return [(value >> bit_index) & 1 for bit_index in range(width)]


def bits_to_index(bits: Iterable[int]) -> int:
    """Decode little-endian qubit bits into an integer basis-state index."""

    value = 0
    for bit_index, bit in enumerate(bits):
        bit_value = int(bit)
        if bit_value not in {0, 1}:
            raise ValueError("QUBO bits must be binary")
        value += bit_value << bit_index
    return value


def decode_candidate_index(bits: Iterable[int], candidate_count: int) -> int | None:
    """Decode bits into a valid candidate index, or ``None`` for unused states."""

    index = bits_to_index(bits)
    return index if 0 <= index < int(candidate_count) else None


def _minmax_loss(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    filled = values.fillna(values.median())
    minimum = float(filled.min())
    maximum = float(filled.max())
    if np.isclose(maximum, minimum):
        return pd.Series(np.zeros(len(filled)), index=series.index)
    return (filled - minimum) / (maximum - minimum)


def _numeric_unit_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    """Return a numeric 0-1 column, filling missing values with a default."""

    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return values.clip(0.0, 1.0)


def _access_point_loss(frame: pd.DataFrame) -> pd.Series:
    """Map access channels to a lower-is-better operational risk loss."""

    if "access point" not in frame.columns:
        return pd.Series(0.5, index=frame.index, dtype=float)

    def loss(raw_value: object) -> float:
        text = str(raw_value or "").strip().lower()
        if not text:
            return 0.5
        if any(token in text for token in ("mobile", "internet", "online", "app", "wallet")):
            return 0.05
        if any(token in text for token in ("card", "atm")):
            return 0.20
        if "agent" in text:
            return 0.35
        if any(token in text for token in ("bank branch", "branch", "office")):
            return 0.60
        return 0.40

    return frame["access point"].map(loss).astype(float).clip(0.0, 1.0)
