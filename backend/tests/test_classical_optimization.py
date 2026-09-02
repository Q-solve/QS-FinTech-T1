"""Correctness tests for the validated classical provider-selection model."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from math import isclose

import pytest

from backend.app.config import CLEANED_DATASET_PATH
from backend.app.optimization.classical_solver import (
    assert_solver_agreement,
    feasible_bitstrings,
    is_exactly_one,
    solve_direct_argmin,
    solve_exhaustive_binary,
)
from backend.app.optimization.data_loader import load_cleaned_dataset
from backend.app.optimization.filters import (
    SPEED_ORDINALS,
    TOTAL_COST_TOLERANCE,
    build_eligible_alternatives,
    encode_speed,
)
from backend.app.optimization.models import (
    Benchmark,
    ObjectiveWeights,
    ServiceAlternative,
)
from backend.app.optimization.normalization import min_max_normalize
from backend.app.optimization.objective import rank_alternatives


@pytest.fixture(scope="module")
def dataset():
    return load_cleaned_dataset(CLEANED_DATASET_PATH)


@pytest.fixture(scope="module")
def cc1_alternatives(dataset):
    return build_eligible_alternatives(
        dataset, period="2025_3Q", corridor="KENTZA", benchmark=Benchmark.CC1
    )


@pytest.fixture(scope="module")
def cc2_alternatives(dataset):
    return build_eligible_alternatives(
        dataset, period="2025_3Q", corridor="KENTZA", benchmark=Benchmark.CC2
    )


def _alternative(
    alternative_id: str,
    firm: str,
    *,
    fee: float,
    fx: float,
    speed: int,
    total: float,
) -> ServiceAlternative:
    return ServiceAlternative(
        alternative_id=alternative_id,
        period="test",
        corridor="TEST",
        firm=firm,
        firm_type="Bank",
        payment_instrument="Cash",
        speed_label="Less than one hour",
        speed_ordinal=speed,
        pickup_method="Cash",
        benchmark=Benchmark.CC1,
        benchmark_amount=100.0,
        benchmark_currency="KES",
        fee_percentage=fee,
        fx_margin=fx,
        total_cost_percentage=total,
        total_cost_residual=0.0,
    )


def test_latest_kentza_has_exactly_14_distinct_alternatives(cc1_alternatives) -> None:
    identities = {
        (
            item.firm,
            item.firm_type,
            item.payment_instrument,
            item.speed_label,
            item.pickup_method,
        )
        for item in cc1_alternatives
    }
    assert len(cc1_alternatives) == 14
    assert len(identities) == 14
    assert len({item.firm for item in cc1_alternatives}) == 11


def test_cc1_and_cc2_select_source_backed_benchmarks(
    cc1_alternatives, cc2_alternatives
) -> None:
    assert {item.benchmark for item in cc1_alternatives} == {Benchmark.CC1}
    assert {item.benchmark_amount for item in cc1_alternatives} == {18_000.0}
    assert {item.benchmark for item in cc2_alternatives} == {Benchmark.CC2}
    assert {item.benchmark_amount for item in cc2_alternatives} == {45_000.0}
    assert {item.benchmark_currency for item in cc1_alternatives} == {"KES"}
    assert {item.benchmark_currency for item in cc2_alternatives} == {"KES"}

    weights = ObjectiveWeights(0.4, 0.3, 0.3)
    for alternatives, expected_benchmark in (
        (cc1_alternatives, Benchmark.CC1),
        (cc2_alternatives, Benchmark.CC2),
    ):
        direct = solve_direct_argmin(alternatives, weights)
        exhaustive = solve_exhaustive_binary(alternatives, weights)
        assert_solver_agreement(direct, exhaustive)
        assert direct.benchmark is expected_benchmark


def test_fee_percentage_is_fee_divided_by_amount(dataset, cc2_alternatives) -> None:
    source = dataset[
        (dataset["PERIOD"] == "2025_3Q")
        & (dataset["CORRIDOR"] == "KENTZA")
        & (dataset["FIRM"] == "Western Union")
    ].iloc[0]
    alternative = next(
        item for item in cc2_alternatives if item.firm == "Western Union"
    )
    expected = float(source["CC2 LCU FEE"]) / float(source["CC2 LCU AMOUNT"]) * 100
    assert alternative.fee_percentage == pytest.approx(expected)
    assert alternative.fee_percentage == pytest.approx(1.15)


@pytest.mark.parametrize("fixture_name", ["cc1_alternatives", "cc2_alternatives"])
def test_total_cost_identity(request, fixture_name) -> None:
    alternatives = request.getfixturevalue(fixture_name)
    for item in alternatives:
        assert item.total_cost_percentage == pytest.approx(
            item.fee_percentage + item.fx_margin, abs=TOTAL_COST_TOLERANCE
        )
        assert abs(item.total_cost_residual) <= TOTAL_COST_TOLERANCE


@pytest.mark.parametrize(("label", "ordinal"), list(SPEED_ORDINALS.items()))
def test_speed_encoding(label: str, ordinal: int) -> None:
    assert encode_speed(label) == ordinal


def test_min_max_normalization() -> None:
    assert min_max_normalize([-2.0, 2.0, 6.0]) == pytest.approx((0.0, 0.5, 1.0))


def test_constant_feature_normalizes_to_neutral_zero() -> None:
    assert min_max_normalize([3.5, 3.5, 3.5]) == (0.0, 0.0, 0.0)


def test_negative_values_are_preserved_and_normalized_relatively(
    cc2_alternatives,
) -> None:
    western_union = next(
        item for item in cc2_alternatives if item.firm == "Western Union"
    )
    assert western_union.fx_margin == -0.69
    margins = [item.fx_margin for item in cc2_alternatives]
    normalized = min_max_normalize(margins)
    index = cc2_alternatives.index(western_union)
    assert normalized[index] == 0.0


@pytest.mark.parametrize(
    "values",
    [
        (-0.1, 0.5, 0.6),
        (1.1, 0.0, -0.1),
        (0.2, 0.2, 0.2),
        (float("nan"), 0.5, 0.5),
    ],
)
def test_invalid_weights_are_rejected(values) -> None:
    with pytest.raises(ValueError, match="weight|Weight"):
        ObjectiveWeights(*values)


def test_weights_accept_small_sum_rounding_tolerance() -> None:
    weights = ObjectiveWeights(0.1, 0.2, 0.7000000001)
    assert weights.speed_weight == pytest.approx(0.7000000001)


def test_deterministic_tie_breaking_order() -> None:
    weights = ObjectiveWeights(0.5, 0.5, 0.0)
    base = _alternative("alt-z", "Zulu", fee=1, fx=1, speed=1, total=5)
    faster = replace(base, alternative_id="alt-faster", speed_ordinal=0)
    alphabetical = replace(base, alternative_id="alt-alpha", firm="Alpha")
    lower_total = replace(base, alternative_id="alt-total", total_cost_percentage=4)
    identifier = replace(alphabetical, alternative_id="alt-000")

    ranked = rank_alternatives(
        [base, faster, alphabetical, lower_total, identifier], weights
    )
    assert [item.alternative.alternative_id for item in ranked] == [
        "alt-total",
        "alt-faster",
        "alt-000",
        "alt-alpha",
        "alt-z",
    ]
    result = solve_direct_argmin(
        [base, faster, alphabetical, lower_total, identifier], weights
    )
    assert result.selected.alternative.alternative_id == "alt-total"
    assert result.tie_information.tie_breaking_applied
    assert result.tie_information.tied_count == 5


def test_exactly_one_binary_constraint() -> None:
    feasible = list(feasible_bitstrings(4))
    assert len(feasible) == 4
    assert all(is_exactly_one(bits) for bits in feasible)
    assert not is_exactly_one((0, 0, 0, 0))
    assert not is_exactly_one((1, 1, 0, 0))
    assert sum(is_exactly_one(bits) for bits in product((0, 1), repeat=4)) == 4


@pytest.mark.parametrize(
    "weights",
    [
        ObjectiveWeights(0.8, 0.1, 0.1),
        ObjectiveWeights(0.1, 0.8, 0.1),
        ObjectiveWeights(0.1, 0.1, 0.8),
    ],
    ids=["fee-focused", "fx-focused", "speed-focused"],
)
def test_classical_solvers_agree_for_preference_scenarios(
    cc2_alternatives, weights
) -> None:
    direct = solve_direct_argmin(cc2_alternatives, weights)
    exhaustive = solve_exhaustive_binary(cc2_alternatives, weights)
    assert_solver_agreement(direct, exhaustive)
    assert direct.eligible_alternatives == 14
    assert direct.selected.weighted_score == pytest.approx(
        exhaustive.selected.weighted_score
    )


def test_ranked_results_are_reproducible(cc2_alternatives) -> None:
    weights = ObjectiveWeights(0.4, 0.3, 0.3)
    first = solve_direct_argmin(cc2_alternatives, weights)
    second = solve_direct_argmin(tuple(reversed(cc2_alternatives)), weights)
    assert [item.alternative.alternative_id for item in first.ranked_alternatives] == [
        item.alternative.alternative_id for item in second.ranked_alternatives
    ]
    assert all(
        isclose(a.weighted_score, b.weighted_score, abs_tol=1e-15)
        for a, b in zip(
            first.ranked_alternatives, second.ranked_alternatives, strict=True
        )
    )
