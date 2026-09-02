"""Shared weighted objective and deterministic ranking logic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cmp_to_key
from math import isclose

from .models import (
    NormalizedComponents,
    ObjectiveWeights,
    RankedAlternative,
    ServiceAlternative,
    TieInformation,
)
from .normalization import normalize_alternatives

SCORE_TIE_TOLERANCE = 1e-12
TIE_BREAK_RULE = (
    "weighted score; then lower total-cost percentage; faster speed ordinal; "
    "provider name; service-alternative identifier"
)


def _compare_ranked(left: RankedAlternative, right: RankedAlternative) -> int:
    """Compare scores with tolerance, then apply the documented tie keys."""

    if not isclose(
        left.weighted_score,
        right.weighted_score,
        rel_tol=0.0,
        abs_tol=SCORE_TIE_TOLERANCE,
    ):
        return -1 if left.weighted_score < right.weighted_score else 1
    left_tie_key = (
        left.alternative.total_cost_percentage,
        left.alternative.speed_ordinal,
        left.alternative.firm,
        left.alternative.alternative_id,
    )
    right_tie_key = (
        right.alternative.total_cost_percentage,
        right.alternative.speed_ordinal,
        right.alternative.firm,
        right.alternative.alternative_id,
    )
    return (left_tie_key > right_tie_key) - (left_tie_key < right_tie_key)


def weighted_score(
    normalized: NormalizedComponents, weights: ObjectiveWeights
) -> float:
    return (
        weights.fee_weight * normalized.fee
        + weights.fx_weight * normalized.fx
        + weights.speed_weight * normalized.speed
    )


def rank_alternatives(
    alternatives: Sequence[ServiceAlternative], weights: ObjectiveWeights
) -> tuple[RankedAlternative, ...]:
    """Score and rank alternatives using the documented stable ordering."""

    normalized = normalize_alternatives(alternatives)
    unranked = [
        RankedAlternative(
            alternative=alternative,
            normalized=components,
            weighted_score=weighted_score(components, weights),
            rank=0,
        )
        for alternative, components in zip(alternatives, normalized, strict=True)
    ]
    ordered = sorted(unranked, key=cmp_to_key(_compare_ranked))
    return tuple(replace(item, rank=index) for index, item in enumerate(ordered, 1))


def describe_winner_tie(ranked: Sequence[RankedAlternative]) -> TieInformation:
    if not ranked:
        raise ValueError("Cannot describe a tie for an empty ranking.")
    best_score = ranked[0].weighted_score
    tied = tuple(
        item.alternative.alternative_id
        for item in ranked
        if isclose(
            item.weighted_score,
            best_score,
            rel_tol=0.0,
            abs_tol=SCORE_TIE_TOLERANCE,
        )
    )
    return TieInformation(
        tied_alternative_ids=tied,
        tied_count=len(tied),
        tie_breaking_applied=len(tied) > 1,
        rule=TIE_BREAK_RULE,
    )
