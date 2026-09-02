"""Strict response contracts exposed by the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class StrictResponseModel(BaseModel):
    """Reject implicit coercion and undocumented response fields."""

    model_config = ConfigDict(strict=True, extra="forbid")


class HealthResponse(StrictResponseModel):
    status: Literal["ok"]
    service: str
    version: str


class CountrySummary(StrictResponseModel):
    code: str
    name: str


class DatasetSummaryResponse(StrictResponseModel):
    record_count: int
    corridor_count: int
    latest_reporting_period: str
    sending_countries: list[CountrySummary]
    receiving_countries: list[CountrySummary]
    kenya_to_tanzania_record_count: int


class ObjectiveWeightsResponse(StrictResponseModel):
    fee: float
    fx_margin: float
    speed: float


class ShowcaseScenarioResponse(StrictResponseModel):
    period: str
    corridor: str
    source: str
    destination: str
    story_destination: str
    benchmark: str
    benchmark_amount: float
    benchmark_currency: str
    alternative_count: int


class ShowcaseAlternativeResponse(StrictResponseModel):
    rank: int
    alternative_id: str
    provider: str
    provider_type: str
    payment_instrument: str
    pickup_method: str
    speed: str
    fee_percentage: float
    fx_margin: float
    total_cost_percentage: float
    weighted_score: float
    selected: bool


class DistributionResponse(StrictResponseModel):
    mean: float | None
    standard_deviation: float | None


class ShowcaseExperimentResponse(StrictResponseModel):
    key: str
    label: str
    mixer: str
    depth: int
    run_count: int
    feasible_probability: DistributionResponse
    leakage_probability: DistributionResponse
    optimal_probability: DistributionResponse
    exact_recovery_count: int
    exact_recovery_rate: float
    modal_feasibility_rate: float
    best_feasible_gap: DistributionResponse
    transpiled_depth: DistributionResponse
    runtime_seconds: DistributionResponse


class ShowcaseProtocolResponse(StrictResponseModel):
    shots: int
    optimizer: str
    maximum_iterations: int
    simulator: str
    matched_seeds_by_depth: dict[str, list[int]]


class ShowcaseResponse(StrictResponseModel):
    title: str
    scenario: ShowcaseScenarioResponse
    weights: ObjectiveWeightsResponse
    recommendation: ShowcaseAlternativeResponse
    alternatives: list[ShowcaseAlternativeResponse]
    experiments: list[ShowcaseExperimentResponse]
    protocol: ShowcaseProtocolResponse
    workflow: list[str]
    limitations: list[str]
    provenance: dict[str, str]
