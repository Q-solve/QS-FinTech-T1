"""Dataset filtering and source-field conversion for provider selection."""

from __future__ import annotations

from hashlib import sha256
from math import isfinite

import pandas as pd

from .models import Benchmark, ServiceAlternative

ALTERNATIVE_FIELDS = (
    "FIRM",
    "FIRM_TYPE",
    "PAYMENT INSTRUMENT",
    "SPEED ACTUAL",
    "PICK-UP METHOD",
)
TOTAL_COST_TOLERANCE = 0.011
SPEED_ORDINALS = {
    "Less than one hour": 0,
    "Same day": 1,
    "Next day": 2,
    "2 days": 3,
    "3-5 days": 4,
}


class EligibilityError(ValueError):
    """Raised when no unambiguous, complete comparison set can be built."""


def encode_speed(speed_label: str) -> int:
    """Encode a source speed band as an ordinal, not a number of hours."""

    try:
        return SPEED_ORDINALS[speed_label]
    except KeyError as exc:
        raise EligibilityError(
            f"Unsupported SPEED ACTUAL value: {speed_label!r}"
        ) from exc


def _alternative_id(values: tuple[str, ...]) -> str:
    canonical = "\x1f".join(values)
    return f"alt-{sha256(canonical.encode('utf-8')).hexdigest()[:12]}"


def _required_text(row: pd.Series, column: str) -> str:
    value = row[column]
    if pd.isna(value):
        raise EligibilityError(f"Eligible row has a missing {column} value.")
    return str(value)


def _required_number(row: pd.Series, column: str) -> float:
    value = row[column]
    if pd.isna(value):
        raise EligibilityError(f"Eligible row has a missing {column} value.")
    number = float(value)
    if not isfinite(number):
        raise EligibilityError(f"Eligible row has a non-finite {column} value.")
    return number


def build_eligible_alternatives(
    dataset: pd.DataFrame,
    *,
    period: str,
    corridor: str,
    benchmark: Benchmark,
) -> tuple[ServiceAlternative, ...]:
    """Filter first, then create one typed decision alternative per source row."""

    filtered = dataset[
        (dataset["PERIOD"] == period) & (dataset["CORRIDOR"] == corridor)
    ]
    if filtered.empty:
        raise EligibilityError(
            f"No records found for PERIOD={period!r} and CORRIDOR={corridor!r}."
        )

    duplicate_mask = filtered.duplicated(subset=list(ALTERNATIVE_FIELDS), keep=False)
    if duplicate_mask.any():
        duplicates = filtered.loc[duplicate_mask, list(ALTERNATIVE_FIELDS)]
        raise EligibilityError(
            "Filtered records do not map one-to-one to service alternatives: "
            f"{len(duplicates)} rows share an alternative definition."
        )

    alternatives: list[ServiceAlternative] = []
    for _, row in filtered.iterrows():
        identity = tuple(_required_text(row, column) for column in ALTERNATIVE_FIELDS)
        amount = _required_number(row, benchmark.amount_column)
        fee = _required_number(row, benchmark.fee_column)
        if amount <= 0:
            raise EligibilityError(
                f"Alternative {_alternative_id(identity)} has a non-positive benchmark amount."
            )
        fee_percentage = fee / amount * 100.0
        fx_margin = _required_number(row, benchmark.fx_margin_column)
        total_cost = _required_number(row, benchmark.total_cost_column)
        residual = total_cost - (fee_percentage + fx_margin)
        if abs(residual) > TOTAL_COST_TOLERANCE:
            raise EligibilityError(
                f"Alternative {_alternative_id(identity)} fails the total-cost identity "
                f"for {benchmark.value}: residual={residual:.6f} percentage points."
            )

        alternatives.append(
            ServiceAlternative(
                alternative_id=_alternative_id(identity),
                period=_required_text(row, "PERIOD"),
                corridor=_required_text(row, "CORRIDOR"),
                firm=identity[0],
                firm_type=identity[1],
                payment_instrument=identity[2],
                speed_label=identity[3],
                speed_ordinal=encode_speed(identity[3]),
                pickup_method=identity[4],
                benchmark=benchmark,
                benchmark_amount=amount,
                benchmark_currency=_required_text(row, benchmark.currency_column),
                fee_percentage=fee_percentage,
                fx_margin=fx_margin,
                total_cost_percentage=total_cost,
                total_cost_residual=residual,
            )
        )

    return tuple(alternatives)
