"""Regression checks for the approved cleaned dataset."""

from fastapi.testclient import TestClient

from backend.app.config import CLEANED_DATASET_PATH
from backend.app.main import app
from backend.app.optimization.data_loader import EXPECTED_COLUMNS, load_cleaned_dataset


client = TestClient(app)


def test_cleaned_dataset_invariants() -> None:
    dataset = load_cleaned_dataset(CLEANED_DATASET_PATH)

    assert dataset.shape == (4026, 30)
    assert tuple(dataset.columns) == EXPECTED_COLUMNS
    assert int(dataset.duplicated(keep=False).sum()) == 0
    assert "2025_3Q" in set(dataset["PERIOD"])
    assert "KENTZA" in set(dataset["CORRIDOR"])


def test_dataset_summary_endpoint() -> None:
    dataset = load_cleaned_dataset(CLEANED_DATASET_PATH)
    expected_kenya_to_tanzania = int(
        (
            (dataset["SOURCE_CODE"] == "KEN")
            & (dataset["DESTINATION_CODE"] == "TZA")
        ).sum()
    )

    response = client.get("/api/dataset/summary")

    assert response.status_code == 200
    assert response.json() == {
        "record_count": 4026,
        "corridor_count": 8,
        "latest_reporting_period": "2025_3Q",
        "sending_countries": [
            {"code": "KEN", "name": "Kenya"},
            {"code": "RWA", "name": "Rwanda"},
            {"code": "TZA", "name": "Tanzania"},
        ],
        "receiving_countries": [
            {"code": "KEN", "name": "Kenya"},
            {"code": "RWA", "name": "Rwanda"},
            {"code": "SSD", "name": "South Sudan"},
            {"code": "TZA", "name": "Tanzania"},
            {"code": "UGA", "name": "Uganda"},
        ],
        "kenya_to_tanzania_record_count": expected_kenya_to_tanzania,
    }
