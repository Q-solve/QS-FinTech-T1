"""Min-max normalization within one eligible comparison set."""

from __future__ import annotations

from collections.abc import Sequence

from .models import NormalizedComponents, ServiceAlternative


def min_max_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize to [0, 1], assigning zero when the feature is constant."""

    if not values:
        raise ValueError("Cannot normalize an empty feature.")
    lower = min(values)
    upper = max(values)
    if upper == lower:
        return tuple(0.0 for _ in values)
    width = upper - lower
    return tuple((value - lower) / width for value in values)


def normalize_alternatives(
    alternatives: Sequence[ServiceAlternative],
) -> tuple[NormalizedComponents, ...]:
    if not alternatives:
        raise ValueError("Cannot normalize an empty alternative set.")
    fees = min_max_normalize([item.fee_percentage for item in alternatives])
    margins = min_max_normalize([item.fx_margin for item in alternatives])
    speeds = min_max_normalize([float(item.speed_ordinal) for item in alternatives])
    return tuple(
        NormalizedComponents(fee=fee, fx=fx, speed=speed)
        for fee, fx, speed in zip(fees, margins, speeds, strict=True)
    )
