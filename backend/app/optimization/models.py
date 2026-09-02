"""Typed contracts for classical remittance-provider selection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite

WEIGHT_SUM_TOLERANCE = 1e-9


class Benchmark(str, Enum):
    """Source-backed World Bank send-amount benchmarks."""

    CC1 = "CC1"
    CC2 = "CC2"

    @property
    def amount_column(self) -> str:
        return f"{self.value} LCU AMOUNT"

    @property
    def fee_column(self) -> str:
        return f"{self.value} LCU FEE"

    @property
    def fx_margin_column(self) -> str:
        return f"{self.value} FX MARGIN"

    @property
    def total_cost_column(self) -> str:
        return f"{self.value} TOTAL COST %"

    @property
    def currency_column(self) -> str:
        return f"{self.value} LCU CODE"


@dataclass(frozen=True)
class ObjectiveWeights:
    """Validated weights for the three non-overlapping objectives."""

    fee_weight: float
    fx_weight: float
    speed_weight: float

    def __post_init__(self) -> None:
        values = (self.fee_weight, self.fx_weight, self.speed_weight)
        if not all(isfinite(value) for value in values):
            raise ValueError("Weights must be finite numbers.")
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Each weight must be between 0 and 1 inclusive.")
        if not isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=WEIGHT_SUM_TOLERANCE):
            raise ValueError(
                "fee_weight, fx_weight, and speed_weight must sum to 1 "
                f"within {WEIGHT_SUM_TOLERANCE:g}."
            )


@dataclass(frozen=True)
class ServiceAlternative:
    """One eligible provider/service observation for a benchmark."""

    alternative_id: str
    period: str
    corridor: str
    firm: str
    firm_type: str
    payment_instrument: str
    speed_label: str
    speed_ordinal: int
    pickup_method: str
    benchmark: Benchmark
    benchmark_amount: float
    benchmark_currency: str
    fee_percentage: float
    fx_margin: float
    total_cost_percentage: float
    total_cost_residual: float


@dataclass(frozen=True)
class NormalizedComponents:
    fee: float
    fx: float
    speed: float


@dataclass(frozen=True)
class RankedAlternative:
    alternative: ServiceAlternative
    normalized: NormalizedComponents
    weighted_score: float
    rank: int


@dataclass(frozen=True)
class TieInformation:
    tied_alternative_ids: tuple[str, ...]
    tied_count: int
    tie_breaking_applied: bool
    rule: str


@dataclass(frozen=True)
class SolverResult:
    selected: RankedAlternative
    benchmark: Benchmark
    weights: ObjectiveWeights
    solver_name: str
    runtime_seconds: float
    eligible_alternatives: int
    tie_information: TieInformation
    ranked_alternatives: tuple[RankedAlternative, ...]
