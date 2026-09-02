"""Exact classical baselines for one-of-N provider/service selection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product
from math import isclose
from time import perf_counter

from .models import (
    ObjectiveWeights,
    RankedAlternative,
    ServiceAlternative,
    SolverResult,
)
from .objective import SCORE_TIE_TOLERANCE, describe_winner_tie, rank_alternatives


def is_exactly_one(bitstring: Sequence[int]) -> bool:
    return all(bit in (0, 1) for bit in bitstring) and sum(bitstring) == 1


def feasible_bitstrings(size: int) -> Iterator[tuple[int, ...]]:
    """Enumerate binary assignments satisfying the exactly-one constraint."""

    if size < 1:
        raise ValueError("At least one decision variable is required.")
    for bitstring in product((0, 1), repeat=size):
        if is_exactly_one(bitstring):
            yield bitstring


def _result(
    *,
    alternatives: Sequence[ServiceAlternative],
    weights: ObjectiveWeights,
    solver_name: str,
    runtime_seconds: float,
    selected_index: int,
    ranked: tuple[RankedAlternative, ...],
) -> SolverResult:
    selected_id = alternatives[selected_index].alternative_id
    selected = next(
        item for item in ranked if item.alternative.alternative_id == selected_id
    )
    return SolverResult(
        selected=selected,
        benchmark=selected.alternative.benchmark,
        weights=weights,
        solver_name=solver_name,
        runtime_seconds=runtime_seconds,
        eligible_alternatives=len(alternatives),
        tie_information=describe_winner_tie(ranked),
        ranked_alternatives=ranked,
    )


def solve_direct_argmin(
    alternatives: Sequence[ServiceAlternative], weights: ObjectiveWeights
) -> SolverResult:
    """Select the first item in the shared deterministic ranking."""

    if not alternatives:
        raise ValueError("At least one eligible alternative is required.")
    started = perf_counter()
    ranked = rank_alternatives(alternatives, weights)
    selected_id = ranked[0].alternative.alternative_id
    selected_index = next(
        index
        for index, item in enumerate(alternatives)
        if item.alternative_id == selected_id
    )
    elapsed = perf_counter() - started
    return _result(
        alternatives=alternatives,
        weights=weights,
        solver_name="direct_weighted_argmin",
        runtime_seconds=elapsed,
        selected_index=selected_index,
        ranked=ranked,
    )


def solve_exhaustive_binary(
    alternatives: Sequence[ServiceAlternative], weights: ObjectiveWeights
) -> SolverResult:
    """Search all 2^N bitstrings and retain only exactly-one assignments."""

    if not alternatives:
        raise ValueError("At least one eligible alternative is required.")
    started = perf_counter()
    ranked = rank_alternatives(alternatives, weights)
    score_by_id = {
        item.alternative.alternative_id: item.weighted_score for item in ranked
    }
    tie_key_by_id = {item.alternative.alternative_id: item.rank for item in ranked}

    best_index: int | None = None
    best_score = float("inf")
    best_tie_rank = len(alternatives) + 1
    for bitstring in product((0, 1), repeat=len(alternatives)):
        if not is_exactly_one(bitstring):
            continue
        selected_index = bitstring.index(1)
        selected_id = alternatives[selected_index].alternative_id
        score = score_by_id[selected_id]
        tie_rank = tie_key_by_id[selected_id]
        if score < best_score - SCORE_TIE_TOLERANCE or (
            isclose(score, best_score, rel_tol=0.0, abs_tol=SCORE_TIE_TOLERANCE)
            and tie_rank < best_tie_rank
        ):
            best_index = selected_index
            best_score = score
            best_tie_rank = tie_rank
    elapsed = perf_counter() - started
    if best_index is None:  # Defensive: N >= 1 always has feasible assignments.
        raise RuntimeError("Exactly-one binary search found no feasible assignment.")
    return _result(
        alternatives=alternatives,
        weights=weights,
        solver_name="exhaustive_exactly_one_binary",
        runtime_seconds=elapsed,
        selected_index=best_index,
        ranked=ranked,
    )


def assert_solver_agreement(first: SolverResult, second: SolverResult) -> None:
    """Raise if two exact solvers disagree on the selected optimum."""

    if (
        first.selected.alternative.alternative_id
        != second.selected.alternative.alternative_id
    ):
        raise AssertionError("Classical solvers selected different alternatives.")
    if not isclose(
        first.selected.weighted_score,
        second.selected.weighted_score,
        rel_tol=0.0,
        abs_tol=SCORE_TIE_TOLERANCE,
    ):
        raise AssertionError("Classical solvers returned different objective values.")
