"""Circuit-based initial states for constraint-preserving QAOA."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, pi, sqrt

from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class WStateValidation:
    qubit_count: int
    norm: float
    expected_one_hot_probability: float
    one_hot_probabilities_by_qiskit_label: dict[str, float]
    maximum_probability_error: float
    infeasible_probability: float


def one_hot_qiskit_label(qubit_count: int, variable_index: int) -> str:
    """Return the x_(n-1)..x_0 label for one excitation at ``x_variable_index``."""

    if qubit_count < 1:
        raise ValueError("qubit_count must be a positive integer.")
    if variable_index < 0 or variable_index >= qubit_count:
        raise ValueError("variable_index must identify an existing qubit.")
    bits_x_order = tuple(
        1 if index == variable_index else 0 for index in range(qubit_count)
    )
    return "".join(str(bit) for bit in reversed(bits_x_order))


def build_uniform_one_hot_state(qubit_count: int) -> QuantumCircuit:
    """Prepare a uniform positive-amplitude W state using O(n) two-qubit gates.

    An excitation starts at x_0. Sequential XX+YY rotations retain amplitude
    1/sqrt(n) on the current qubit and pass the remaining amplitude onward.
    ``beta=pi/2`` compensates Qiskit's swap phase, yielding positive amplitudes.
    """

    if qubit_count < 1:
        raise ValueError("qubit_count must be a positive integer.")
    circuit = QuantumCircuit(qubit_count, name=f"W_{qubit_count}")
    circuit.x(0)
    for left in range(qubit_count - 1):
        remaining = qubit_count - left
        angle = 2.0 * acos(1.0 / sqrt(remaining))
        circuit.append(XXPlusYYGate(angle, beta=pi / 2.0), [left, left + 1])
    return circuit


def validate_uniform_one_hot_state(qubit_count: int) -> WStateValidation:
    circuit = build_uniform_one_hot_state(qubit_count)
    state = Statevector.from_instruction(circuit)
    probabilities = state.probabilities_dict()
    expected = 1.0 / qubit_count
    one_hot = {
        one_hot_qiskit_label(qubit_count, index): float(
            probabilities.get(one_hot_qiskit_label(qubit_count, index), 0.0)
        )
        for index in range(qubit_count)
    }
    infeasible = sum(
        float(probability)
        for label, probability in probabilities.items()
        if label.count("1") != 1
    )
    return WStateValidation(
        qubit_count=qubit_count,
        norm=float(state.inner(state).real),
        expected_one_hot_probability=expected,
        one_hot_probabilities_by_qiskit_label=one_hot,
        maximum_probability_error=max(
            abs(probability - expected) for probability in one_hot.values()
        ),
        infeasible_probability=infeasible,
    )
