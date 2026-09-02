"""Application configuration with repository-relative data paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "remittance_east_africa_clean.csv"
)

APP_NAME = "Remit-Q API"
APP_VERSION = "0.1.0"
API_PREFIX = "/api"

CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
