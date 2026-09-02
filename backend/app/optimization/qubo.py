"""QUBO construction for the validated one-of-N provider selection model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from .models import ObjectiveWeights, RankedAlternative, ServiceAlternative
from .objective import rank_alternatives

EXACTLY_ONE_CONSTRAINT = "select_exactly_one"
DEFAULT_QUBO_PENALTY = 2.0


@dataclass(frozen=True)
class QuboRepresentation:
    """Solver-independent, binary-reduced coefficients for inspection."""

    variable_order: tuple[str, ...]
    alternative_ids: tuple[str, ...]
    scores: tuple[float, ...]
    penalty: float
    constant_offset: float
    linear_coefficients: dict[str, float]
    quadratic_coefficients: dict[tuple[str, str], float]
    upper_triangular_matrix: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class QuboModel:
    """The shared alternatives plus constrained and converted Qiskit models."""

    alternatives: tuple[ServiceAlternative, ...]
    ranked_alternatives: tuple[RankedAlternative, ...]
    representation: QuboRepresentation
    constrained_problem: QuadraticProgram
    qubo_problem: QuadraticProgram


def validate_penalty(penalty: float) -> float:
    """Return a finite positive penalty or raise a clear validation error."""

    if isinstance(penalty, bool) or not isinstance(penalty, Real):
        raise TypeError("QUBO penalty must be a finite positive number.")
    validated = float(penalty)
    if not isfinite(validated) or validated <= 0.0:
        raise ValueError("QUBO penalty must be a finite positive number.")
    return validated


def minimum_safe_penalty_threshold(scores: Sequence[float]) -> float:
    """Return the strict lower threshold P must exceed for this one-hot QUBO.

    Normalized weighted scores are nonnegative. The best feasible energy is
    min(scores), while the all-zero state has energy P and every state with two
    or more selected variables has energy at least P. Thus P > min(scores) is
    sufficient; equality can create an infeasible tie.
    """

    if not scores:
        raise ValueError("At least one weighted score is required.")
    values = tuple(float(score) for score in scores)
    if not all(isfinite(score) and 0.0 <= score <= 1.0 for score in values):
        raise ValueError("Weighted scores must be finite values in [0, 1].")
    return min(values)


def is_penalty_sufficient(scores: Sequence[float], penalty: float) -> bool:
    return validate_penalty(penalty) > minimum_safe_penalty_threshold(scores)


def _stable_alternatives(
    alternatives: Sequence[ServiceAlternative],
) -> tuple[ServiceAlternative, ...]:
    if not alternatives:
        raise ValueError("At least one eligible alternative is required.")
    ordered = tuple(sorted(alternatives, key=lambda item: item.alternative_id))
    identifiers = tuple(item.alternative_id for item in ordered)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Service-alternative identifiers must be unique.")
    return ordered


def build_constrained_quadratic_program(
    alternatives: Sequence[ServiceAlternative], weights: ObjectiveWeights
) -> tuple[
    QuadraticProgram, tuple[ServiceAlternative, ...], tuple[RankedAlternative, ...]
]:
    """Build the linear objective and exactly-one equality before conversion."""

    ordered = _stable_alternatives(alternatives)
    ranked = rank_alternatives(ordered, weights)
    score_by_id = {
        item.alternative.alternative_id: item.weighted_score for item in ranked
    }
    problem = QuadraticProgram(name="remittance_provider_selection")
    variable_names = tuple(f"x_{index}" for index in range(len(ordered)))
    for variable_name in variable_names:
        problem.binary_var(name=variable_name)
    problem.minimize(
        linear={
            variable_name: score_by_id[alternative.alternative_id]
            for variable_name, alternative in zip(variable_names, ordered, strict=True)
        }
    )
    problem.linear_constraint(
        linear={variable_name: 1.0 for variable_name in variable_names},
        sense="==",
        rhs=1.0,
        name=EXACTLY_ONE_CONSTRAINT,
    )
    return problem, ordered, ranked


def _representation(
    *,
    alternatives: tuple[ServiceAlternative, ...],
    ranked: tuple[RankedAlternative, ...],
    penalty: float,
) -> QuboRepresentation:
    variable_order = tuple(f"x_{index}" for index in range(len(alternatives)))
    score_by_id = {
        item.alternative.alternative_id: item.weighted_score for item in ranked
    }
    scores = tuple(score_by_id[item.alternative_id] for item in alternatives)
    linear = {
        variable_name: score - penalty
        for variable_name, score in zip(variable_order, scores, strict=True)
    }
    quadratic = {
        (variable_order[left], variable_order[right]): 2.0 * penalty
        for left in range(len(variable_order))
        for right in range(left + 1, len(variable_order))
    }
    matrix_rows: list[tuple[float, ...]] = []
    for row, variable_name in enumerate(variable_order):
        matrix_rows.append(
            tuple(
                linear[variable_name]
                if column == row
                else quadratic[(variable_name, variable_order[column])]
                if column > row
                else 0.0
                for column in range(len(variable_order))
            )
        )
    return QuboRepresentation(
        variable_order=variable_order,
        alternative_ids=tuple(item.alternative_id for item in alternatives),
        scores=scores,
        penalty=penalty,
        constant_offset=penalty,
        linear_coefficients=linear,
        quadratic_coefficients=quadratic,
        upper_triangular_matrix=tuple(matrix_rows),
    )


def build_qubo_model(
    alternatives: Sequence[ServiceAlternative],
    weights: ObjectiveWeights,
    *,
    penalty: float = DEFAULT_QUBO_PENALTY,
) -> QuboModel:
    """Build and explicitly convert the shared constrained model to a QUBO."""

    validated_penalty = validate_penalty(penalty)
    constrained, ordered, ranked = build_constrained_quadratic_program(
        alternatives, weights
    )
    qubo = QuadraticProgramToQubo(penalty=validated_penalty).convert(constrained)
    return QuboModel(
        alternatives=ordered,
        ranked_alternatives=ranked,
        representation=_representation(
            alternatives=ordered,
            ranked=ranked,
            penalty=validated_penalty,
        ),
        constrained_problem=constrained,
        qubo_problem=qubo,
    )


def qubo_energy(model: QuboModel, bit_values: Sequence[int]) -> float:
    """Evaluate conventional QUBO coefficients without invoking a solver."""

    bits = tuple(bit_values)
    variable_order = model.representation.variable_order
    if len(bits) != len(variable_order) or any(bit not in (0, 1) for bit in bits):
        raise ValueError(
            f"Expected {len(variable_order)} binary values in x_0..x_n order."
        )
    energy = model.representation.constant_offset
    for variable_name, bit in zip(variable_order, bits, strict=True):
        energy += model.representation.linear_coefficients[variable_name] * bit
    for (
        left,
        right,
    ), coefficient in model.representation.quadratic_coefficients.items():
        left_index = int(left.removeprefix("x_"))
        right_index = int(right.removeprefix("x_"))
        energy += coefficient * bits[left_index] * bits[right_index]
    return energy
