"""Ring XY mixer construction and independent constraint-preservation checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cache

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from .initial_states import build_uniform_one_hot_state, validate_uniform_one_hot_state

VALIDATION_ANGLES = (0.0, 0.17, 0.63, 1.2)
CONSTRAINT_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ConstraintPreservationValidation:
    qubit_count: int
    connectivity: tuple[tuple[int, int], ...]
    mixer_hermiticity_error: float
    number_commutator_error: float
    mixer_leakage_by_angle: dict[str, float]
    maximum_mixer_leakage: float
    initial_state_validation: dict[str, object]


def ring_connectivity(qubit_count: int) -> tuple[tuple[int, int], ...]:
    if qubit_count < 3:
        raise ValueError("The periodic XY ring requires at least three qubits.")
    return tuple((index, (index + 1) % qubit_count) for index in range(qubit_count))


def build_xy_ring_hamiltonian(qubit_count: int) -> SparsePauliOp:
    """Return H_M = sum_(i,j in ring) (X_i X_j + Y_i Y_j)."""

    terms = []
    for left, right in ring_connectivity(qubit_count):
        terms.extend((("XX", [left, right], 1.0), ("YY", [left, right], 1.0)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=qubit_count).simplify()


def build_xy_ring_mixer_circuit(qubit_count: int) -> QuantumCircuit:
    """First-order product formula for exp(-i beta H_M).

    Each XX+YY edge evolution exactly preserves excitation number. Their
    ordered product therefore preserves Hamming weight even though adjacent
    ring-edge Hamiltonian terms need not commute.
    """

    beta = Parameter("xy_beta")
    circuit = QuantumCircuit(qubit_count, name="xy_ring_mixer")
    for left, right in ring_connectivity(qubit_count):
        # XXPlusYYGate(theta) = exp[-i theta/4 (XX + YY)].
        circuit.append(XXPlusYYGate(4.0 * beta), [left, right])
    return circuit


def excitation_number_operator(qubit_count: int) -> SparsePauliOp:
    """Return N = sum_i (I - Z_i) / 2, whose eigenvalue is Hamming weight."""

    if qubit_count < 1:
        raise ValueError("qubit_count must be a positive integer.")
    terms = [("I", [], qubit_count / 2.0)]
    terms.extend(("Z", [index], -0.5) for index in range(qubit_count))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=qubit_count).simplify()


def _maximum_coefficient(operator: SparsePauliOp) -> float:
    simplified = operator.simplify(atol=CONSTRAINT_TOLERANCE)
    return max((abs(value) for value in simplified.coeffs), default=0.0)


@cache
def validate_constraint_preserving_design(
    qubit_count: int,
) -> ConstraintPreservationValidation:
    """Validate W initialization, Hermiticity, [H_M,N]=0, and mixer leakage."""

    connectivity = ring_connectivity(qubit_count)
    hamiltonian = build_xy_ring_hamiltonian(qubit_count)
    hermiticity_error = _maximum_coefficient(hamiltonian - hamiltonian.adjoint())
    number = excitation_number_operator(qubit_count)
    commutator = hamiltonian.compose(number) - number.compose(hamiltonian)
    commutator_error = _maximum_coefficient(commutator)

    initial_circuit = build_uniform_one_hot_state(qubit_count)
    mixer_circuit = build_xy_ring_mixer_circuit(qubit_count)
    beta = next(iter(mixer_circuit.parameters))
    initial_state = Statevector.from_instruction(initial_circuit)
    leakage_by_angle: dict[str, float] = {}
    for angle in VALIDATION_ANGLES:
        evolved = initial_state.evolve(mixer_circuit.assign_parameters({beta: angle}))
        leakage_by_angle[f"{angle:g}"] = sum(
            float(probability)
            for label, probability in evolved.probabilities_dict().items()
            if label.count("1") != 1
        )
    validation = validate_uniform_one_hot_state(qubit_count)
    result = ConstraintPreservationValidation(
        qubit_count=qubit_count,
        connectivity=connectivity,
        mixer_hermiticity_error=hermiticity_error,
        number_commutator_error=commutator_error,
        mixer_leakage_by_angle=leakage_by_angle,
        maximum_mixer_leakage=max(leakage_by_angle.values()),
        initial_state_validation=asdict(validation),
    )
    if (
        abs(validation.norm - 1.0) > CONSTRAINT_TOLERANCE
        or validation.maximum_probability_error > CONSTRAINT_TOLERANCE
        or validation.infeasible_probability > CONSTRAINT_TOLERANCE
        or hermiticity_error > CONSTRAINT_TOLERANCE
        or commutator_error > CONSTRAINT_TOLERANCE
        or result.maximum_mixer_leakage > CONSTRAINT_TOLERANCE
    ):
        raise AssertionError(
            "Constraint-preserving mixer or W-state validation failed."
        )
    return result
