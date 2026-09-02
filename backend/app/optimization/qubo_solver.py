"""Exact correctness oracles for the provider-selection QUBO."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isclose

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from .classical_solver import is_exactly_one
from .models import ServiceAlternative
from .qubo import QuboModel, qubo_energy

ENERGY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class QuboSolution:
    """An exact solution decoded in documented x_0-through-x_n order."""

    bit_values: tuple[int, ...]
    bit_string: str
    qiskit_state_label: str | None
    energy: float
    feasible: bool
    selected_alternative: ServiceAlternative | None
    original_weighted_score: float | None
    solver_name: str


@dataclass(frozen=True)
class EnergyValidation:
    states_checked: int
    maximum_absolute_difference: float


def _solution(
    model: QuboModel,
    bits: tuple[int, ...],
    *,
    solver_name: str,
    qiskit_state_label: str | None = None,
) -> QuboSolution:
    feasible = is_exactly_one(bits)
    selected_index = bits.index(1) if feasible else None
    selected = (
        model.alternatives[selected_index] if selected_index is not None else None
    )
    original_score = (
        model.representation.scores[selected_index]
        if selected_index is not None
        else None
    )
    return QuboSolution(
        bit_values=bits,
        bit_string="".join(str(bit) for bit in bits),
        qiskit_state_label=qiskit_state_label,
        energy=qubo_energy(model, bits),
        feasible=feasible,
        selected_alternative=selected,
        original_weighted_score=original_score,
        solver_name=solver_name,
    )


def validate_all_qubo_energies(
    model: QuboModel, *, tolerance: float = ENERGY_TOLERANCE
) -> EnergyValidation:
    """Compare independent and Qiskit objective energies for every state."""

    maximum_difference = 0.0
    state_count = 0
    for bits in product((0, 1), repeat=len(model.alternatives)):
        independent = qubo_energy(model, bits)
        qiskit_energy = float(model.qubo_problem.objective.evaluate(list(bits)))
        difference = abs(independent - qiskit_energy)
        maximum_difference = max(maximum_difference, difference)
        state_count += 1
        if difference > tolerance:
            raise AssertionError(
                "Independent and Qiskit QUBO energies disagree for "
                f"{bits}: {independent} != {qiskit_energy}."
            )
    return EnergyValidation(state_count, maximum_difference)


def solve_qubo_by_enumeration(model: QuboModel) -> QuboSolution:
    """Enumerate every unconstrained binary state and minimize QUBO energy."""

    best_bits: tuple[int, ...] | None = None
    best_energy = float("inf")
    classical_rank_by_id = {
        item.alternative.alternative_id: item.rank for item in model.ranked_alternatives
    }
    for bits in product((0, 1), repeat=len(model.alternatives)):
        energy = qubo_energy(model, bits)
        if energy < best_energy - ENERGY_TOLERANCE:
            best_bits = bits
            best_energy = energy
        elif (
            best_bits is not None
            and isclose(energy, best_energy, rel_tol=0.0, abs_tol=ENERGY_TOLERANCE)
            and is_exactly_one(bits)
            and is_exactly_one(best_bits)
        ):
            candidate_id = model.alternatives[bits.index(1)].alternative_id
            incumbent_id = model.alternatives[best_bits.index(1)].alternative_id
            if classical_rank_by_id[candidate_id] < classical_rank_by_id[incumbent_id]:
                best_bits = bits
    if best_bits is None:  # Defensive: a nonempty variable set is required.
        raise RuntimeError("QUBO enumeration produced no binary states.")
    return _solution(model, best_bits, solver_name="independent_qubo_enumeration")


def solve_qubo_with_exact_eigensolver(model: QuboModel) -> QuboSolution:
    """Use NumPyMinimumEigensolver as a correctness oracle, not performance."""

    optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = optimizer.solve(model.qubo_problem)
    bits = tuple(round(value) for value in result.x)

    state_label: str | None = None
    minimum_result = result.min_eigen_solver_result
    if minimum_result is not None and minimum_result.eigenstate is not None:
        amplitudes = minimum_result.eigenstate.to_dict()
        state_label = str(max(amplitudes, key=lambda label: abs(amplitudes[label])))

    solution = _solution(
        model,
        bits,
        solver_name="qiskit_numpy_minimum_eigensolver",
        qiskit_state_label=state_label,
    )
    if not isclose(
        solution.energy,
        float(result.fval),
        rel_tol=0.0,
        abs_tol=ENERGY_TOLERANCE,
    ):
        raise AssertionError("Decoded exact-eigensolver energy does not match Qiskit.")
    return solution


def to_ising(model: QuboModel):
    """Return Qiskit's Ising operator and objective offset for inspection."""

    return model.qubo_problem.to_ising()
