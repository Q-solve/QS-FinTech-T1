"""Hard policy constraints for QKash candidate eligibility."""

from __future__ import annotations

# Imports.
from dataclasses import dataclass

import pandas as pd

from qkash.data import source_column_available


# Variable descriptions.
# DEFAULT_MAX_TOTAL_COST_PCT is permissive so the default app run keeps historical coverage.
DEFAULT_MAX_TOTAL_COST_PCT = 100.0

# DEFAULT_MAX_SETTLEMENT_DAYS allows all standard RPW speed labels unless the user tightens it.
DEFAULT_MAX_SETTLEMENT_DAYS = 5.0

# DEFAULT_MIN_COVERAGE_SCORE keeps the constraint inactive until the user raises the threshold.
DEFAULT_MIN_COVERAGE_SCORE = 0.0

# DEFAULT_MAX_PROVIDER_SHARE is the no-op provider concentration limit.
DEFAULT_MAX_PROVIDER_SHARE = 1.0


@dataclass(frozen=True)
class PolicyConstraints:
    """Hard eligibility constraints applied before Pareto pruning."""

    max_total_cost_pct: float | None = DEFAULT_MAX_TOTAL_COST_PCT
    max_settlement_days: float | None = DEFAULT_MAX_SETTLEMENT_DAYS
    require_transparency: bool = False
    min_coverage_score: float | None = DEFAULT_MIN_COVERAGE_SCORE
    allowed_access_points: tuple[str, ...] = ()
    max_provider_share: float = DEFAULT_MAX_PROVIDER_SHARE


def apply_policy_constraints(
    candidates: pd.DataFrame,
    policy: PolicyConstraints,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Remove services that violate hard policy constraints."""

    if candidates.empty:
        return candidates.copy(), _report(policy, 0, {}, {}, pd.Series(dtype=bool))

    eligible = candidates.copy()
    keep = pd.Series(True, index=eligible.index)
    removals: dict[str, int] = {}
    unsupported: dict[str, str] = {}

    if policy.max_total_cost_pct is not None:
        condition = pd.to_numeric(eligible["total_cost_pct"], errors="coerce").fillna(float("inf"))
        failed = condition > float(policy.max_total_cost_pct)
        keep &= ~failed
        removals["total_cost_policy"] = int(failed.sum())

    if policy.max_settlement_days is not None:
        condition = pd.to_numeric(eligible["settlement_days"], errors="coerce").fillna(float("inf"))
        failed = condition > float(policy.max_settlement_days)
        keep &= ~failed
        removals["settlement_time_policy"] = int(failed.sum())

    if policy.require_transparency:
        if source_column_available(eligible, "transparent"):
            transparent_score = pd.to_numeric(
                eligible["transparent_score"],
                errors="coerce",
            ).fillna(0.0)
            failed = transparent_score < 1.0
            keep &= ~failed
            removals["transparency_policy"] = int(failed.sum())
        else:
            unsupported["transparency_policy"] = "transparent is not available in the source data"

    if policy.min_coverage_score is not None and float(policy.min_coverage_score) > 0.0:
        if source_column_available(eligible, "receiving network coverage"):
            coverage_score = pd.to_numeric(eligible["coverage_score"], errors="coerce").fillna(0.0)
            failed = coverage_score < float(policy.min_coverage_score)
            keep &= ~failed
            removals["network_coverage_policy"] = int(failed.sum())
        else:
            unsupported["network_coverage_policy"] = (
                "receiving network coverage is not available in the source data"
            )

    if policy.allowed_access_points:
        if source_column_available(eligible, "access point"):
            failed = ~_access_point_allowed(eligible["access point"], policy.allowed_access_points)
            keep &= ~failed
            removals["access_point_policy"] = int(failed.sum())
        else:
            unsupported["access_point_policy"] = "access point is not available in the source data"

    filtered = eligible.loc[keep].reset_index(drop=True)
    return filtered, _report(policy, len(candidates), removals, unsupported, keep)


def _access_point_allowed(series: pd.Series, allowed_values: tuple[str, ...]) -> pd.Series:
    """Return rows whose comma-separated access-point tokens match the allowed set."""

    allowed = {value.strip().lower() for value in allowed_values if value.strip()}
    if not allowed:
        return pd.Series(True, index=series.index)

    def matches(raw_value: object) -> bool:
        tokens = {token.strip().lower() for token in str(raw_value or "").split(",")}
        return bool(tokens & allowed)

    return series.fillna("").astype(str).map(matches)


def _report(
    policy: PolicyConstraints,
    input_rows: int,
    removals: dict[str, int],
    unsupported: dict[str, str],
    keep: pd.Series,
) -> dict[str, object]:
    """Build a UI-safe policy report without exposing row data."""

    eligible_rows = int(keep.sum()) if not keep.empty else 0
    return {
        "input_rows": int(input_rows),
        "eligible_rows": eligible_rows,
        "removed_rows": max(int(input_rows) - eligible_rows, 0),
        "removals": removals,
        "unsupported_constraints": unsupported,
        "policy": {
            "max_total_cost_pct": policy.max_total_cost_pct,
            "max_settlement_days": policy.max_settlement_days,
            "require_transparency": policy.require_transparency,
            "min_coverage_score": policy.min_coverage_score,
            "allowed_access_points": list(policy.allowed_access_points),
            "max_provider_share": policy.max_provider_share,
        },
    }
