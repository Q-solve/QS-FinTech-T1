"""Initial read-only API routes."""

import json

from fastapi import APIRouter, HTTPException, status

from ..config import API_PREFIX, APP_NAME, APP_VERSION
from ..optimization.data_loader import DatasetLoadError, load_cleaned_dataset
from ..optimization.preprocessing import build_dataset_summary
from ..schemas import DatasetSummaryResponse, HealthResponse, ShowcaseResponse
from ..showcase import build_showcase

router = APIRouter(prefix=API_PREFIX)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=APP_NAME, version=APP_VERSION)


@router.get("/dataset/summary", response_model=DatasetSummaryResponse)
def dataset_summary() -> DatasetSummaryResponse:
    try:
        dataset = load_cleaned_dataset()
        return build_dataset_summary(dataset)
    except (DatasetLoadError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The cleaned dataset is unavailable or invalid.",
        ) from exc


@router.get("/showcase", response_model=ShowcaseResponse)
def showcase() -> ShowcaseResponse:
    """Return the fixed, validated demonstration and stored experiment summaries."""

    try:
        return build_showcase()
    except (
        DatasetLoadError,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The validated showcase artifacts are unavailable or inconsistent.",
        ) from exc
