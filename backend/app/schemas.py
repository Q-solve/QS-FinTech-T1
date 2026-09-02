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
