"""API smoke tests."""

import json

import pytest
from fastapi.testclient import TestClient

from backend.app import showcase
from backend.app.config import QAOA_COMPARISON_PATH
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Remit-Q API",
        "version": "0.1.0",
    }


def test_showcase_endpoint_exposes_validated_results() -> None:
    response = client.get("/api/showcase")

    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario"] == {
        "period": "2025_3Q",
        "corridor": "KENTZA",
        "source": "Kenya",
        "destination": "Tanzania",
        "story_destination": "Zanzibar (represented by Tanzania in the dataset)",
        "benchmark": "CC2",
        "benchmark_amount": 45000.0,
        "benchmark_currency": "KES",
        "alternative_count": 14,
    }
    assert len(payload["alternatives"]) == 14
    assert sum(item["selected"] for item in payload["alternatives"]) == 1
    experiments = {item["key"]: item for item in payload["experiments"]}
    assert set(experiments) == {
        "standard_x_p1",
        "constraint_preserving_xy_p1",
        "standard_x_p2",
        "constraint_preserving_xy_p2",
    }
    assert (
        experiments["constraint_preserving_xy_p1"]["feasible_probability"]["mean"]
        == 1.0
    )
    assert experiments["standard_x_p1"]["feasible_probability"]["mean"] < 1.0
    assert payload["protocol"]["shots"] == 2_048
    assert payload["protocol"]["matched_seeds_by_depth"]["1"] == [11, 29, 47, 71, 97]
    assert "quantum advantage" in payload["limitations"][0]


def test_showcase_rejects_results_from_a_different_dataset(
    tmp_path, monkeypatch
) -> None:
    changed_dataset = tmp_path / "changed.csv"
    changed_dataset.write_text("not-the-validated-dataset\n", encoding="utf-8")
    comparison = json.loads(QAOA_COMPARISON_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(showcase, "CLEANED_DATASET_PATH", changed_dataset)

    try:
        showcase._validate_comparison(comparison, [])
    except ValueError as exc:
        assert "different processed dataset" in str(exc)
    else:
        raise AssertionError("A mismatched dataset hash must be rejected.")


def test_showcase_rejects_changed_source_experiment() -> None:
    comparison = json.loads(QAOA_COMPARISON_PATH.read_text(encoding="utf-8"))
    comparison["input_artifacts"]["standard_x_p1"]["sha256"] = "0" * 64
    objective_inputs = comparison["comparison_basis"][
        "objective_inputs_in_variable_order"
    ]

    with pytest.raises(ValueError, match="input artifact .* has changed"):
        showcase._validate_comparison(comparison, objective_inputs)
