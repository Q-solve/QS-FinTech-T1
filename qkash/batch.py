"""Batch-transfer planning helpers for QKash research mode."""

from __future__ import annotations

# Imports.
from collections import Counter
from dataclasses import dataclass
import itertools
import math

import pandas as pd


# Variable descriptions.
# DEFAULT_BATCH_PLAN_CAP keeps the plan-level QUBO small enough for local QAOA experiments.
DEFAULT_BATCH_PLAN_CAP = 64

# DEFAULT_BATCH_CANDIDATE_CAP limits combinatorial growth before plan enumeration.
DEFAULT_BATCH_CANDIDATE_CAP = 10


@dataclass(frozen=True)
class BatchSettings:
    """Controls for turning one corridor request into a small batch experiment."""

    transfer_count: int = 1
    max_provider_share: float = 1.0
    candidate_cap: int = DEFAULT_BATCH_CANDIDATE_CAP
    plan_cap: int = DEFAULT_BATCH_PLAN_CAP


def build_batch_plans(
    service_candidates: pd.DataFrame,
    settings: BatchSettings,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build feasible batch plans from candidate services.

    The current UI batch mode repeats the same source/destination/amount demand
    multiple times.  Each plan assigns every transfer in that batch to one
    eligible service while respecting the provider concentration limit.
    """

    transfer_count = max(int(settings.transfer_count), 1)
    if transfer_count <= 1 or service_candidates.empty:
        rows = service_candidates.copy().reset_index(drop=True)
        return rows, _report(
            input_candidates=len(service_candidates),
            enumerated_plans=len(rows),
            kept_plans=len(rows),
            transfer_count=transfer_count,
            max_provider_share=settings.max_provider_share,
            provider_limit=1,
        )

    candidate_cap = max(int(settings.candidate_cap), 1)
    plan_cap = max(int(settings.plan_cap), 1)
    candidates = (
        service_candidates.sort_values("weighted_score", kind="mergesort")
        .head(candidate_cap)
        .reset_index(drop=True)
    )
    provider_limit = max(1, math.floor(float(settings.max_provider_share) * transfer_count))

    plan_rows: list[dict[str, object]] = []
    enumerated = 0
    for assignment in itertools.combinations_with_replacement(range(len(candidates)), transfer_count):
        enumerated += 1
        selected = candidates.iloc[list(assignment)]
        provider_counts = Counter(selected["firm"].astype(str))
        if max(provider_counts.values(), default=0) > provider_limit:
            continue

        representative = selected.sort_values("weighted_score", kind="mergesort").iloc[0]
        plan_rows.append(_plan_row(selected, representative, assignment, provider_counts))

    plans = pd.DataFrame(plan_rows)
    if not plans.empty:
        plans = (
            plans.sort_values("weighted_score", kind="mergesort")
            .head(plan_cap)
            .reset_index(drop=True)
        )

    return plans, _report(
        input_candidates=len(candidates),
        enumerated_plans=enumerated,
        kept_plans=len(plans),
        transfer_count=transfer_count,
        max_provider_share=settings.max_provider_share,
        provider_limit=provider_limit,
    )


def _plan_row(
    selected: pd.DataFrame,
    representative: pd.Series,
    assignment: tuple[int, ...],
    provider_counts: Counter,
) -> dict[str, object]:
    """Aggregate one batch assignment into a QUBO candidate row."""

    plan = representative.to_dict()
    selected_firms = [str(value) for value in selected["firm"].tolist()]
    selected_methods = [
        f"{row['firm']} | {row['payment instrument']} | {row['pickup method']}"
        for _, row in selected.iterrows()
    ]
    plan.update(
        {
            "firm": "; ".join(f"{firm} x{count}" for firm, count in sorted(provider_counts.items())),
            "payment instrument": "Batch plan",
            "pickup method": "Mixed receiving methods",
            "speed actual": "Batch average",
            "batch_assignment": tuple(int(index) for index in assignment),
            "batch_transfer_count": len(assignment),
            "batch_selected_firms": "; ".join(selected_firms),
            "batch_selected_services": " || ".join(selected_methods),
            "batch_provider_counts": dict(provider_counts),
            "batch_mode": True,
            "fee_lcu": float(selected["fee_lcu"].mean()),
            "fx_margin": float(selected["fx_margin"].mean()),
            "total_cost_pct": float(selected["total_cost_pct"].mean()),
            "settlement_days": float(selected["settlement_days"].mean()),
            "transaction_fee_loss": float(selected["transaction_fee_loss"].mean()),
            "fx_spread_loss": float(selected["fx_spread_loss"].mean()),
            "time_loss": float(selected["time_loss"].mean()),
            "risk_loss": float(selected["risk_loss"].mean()),
            "weighted_score": float(selected["weighted_score"].mean()),
        }
    )
    return plan


def _report(
    input_candidates: int,
    enumerated_plans: int,
    kept_plans: int,
    transfer_count: int,
    max_provider_share: float,
    provider_limit: int,
) -> dict[str, object]:
    """Return UI-safe batch-enumeration metadata."""

    return {
        "input_candidates": int(input_candidates),
        "enumerated_plans": int(enumerated_plans),
        "kept_plans": int(kept_plans),
        "transfer_count": int(transfer_count),
        "max_provider_share": float(max_provider_share),
        "provider_limit": int(provider_limit),
    }
