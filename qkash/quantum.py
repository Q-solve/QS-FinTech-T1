"""Qiskit/Qiskit Aer QAOA and quantum backend execution for QKash."""

from __future__ import annotations

# Imports.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

try:
    from qbraid import QbraidProvider
except Exception:  # pragma: no cover - dependency availability is environment specific
    QbraidProvider = None

from .benchmark import sample_from_bits
from .qbraid_bridge import get_qbraid_api_key, load_qbraid_config
from .scoring import QuboModel


# Variable descriptions.
# DEFAULT_QAOA_REPS is the default QAOA depth, p.
DEFAULT_QAOA_REPS = 1

# DEFAULT_SHOTS controls how many bitstrings are sampled from the final circuit.
DEFAULT_SHOTS = 512

# DEFAULT_MAX_QUBITS caps the QAOA search space for local simulation and qBraid.
DEFAULT_MAX_QUBITS = 20


@dataclass(frozen=True)
class QuantumBackendResult:
    """Execution output returned by any quantum backend."""

    counts: dict[str, int]
    backend_name: str
    runtime_s: float
    status: str = "ok"
    note: str = ""
    job_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class QuantumBackend(ABC):
    """Base class for final optimized circuit execution backends."""

    name = "quantum-backend"

    @abstractmethod
    def execute(self, circuit: QuantumCircuit, shots: int) -> QuantumBackendResult:
        """Execute a measured quantum circuit and return measurement counts."""


class LocalAerBackend(QuantumBackend):
    """Run the optimized measured circuit with Qiskit Aer locally."""

    name = "LocalAerBackend"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.simulator = AerSimulator(seed_simulator=seed)

    def execute(self, circuit: QuantumCircuit, shots: int) -> QuantumBackendResult:
        started = time.perf_counter()
        result = self.simulator.run(
            circuit,
            shots=max(int(shots), 1),
            seed_simulator=self.seed,
        ).result()
        counts = _normalize_counts(result.get_counts(circuit), circuit.num_qubits)
        return QuantumBackendResult(
            counts=counts,
            backend_name=self.name,
            runtime_s=time.perf_counter() - started,
            note="Executed final optimized circuit with Qiskit Aer.",
        )


class QBraidBackend(QuantumBackend):
    """Submit only the final optimized measured circuit through qBraid."""

    name = "QBraidBackend"

    def __init__(self, device_id: str | None = None, timeout_s: int = 300) -> None:
        config = load_qbraid_config()
        self.api_key = get_qbraid_api_key()
        self.device_id = device_id or config.device_id
        self.timeout_s = int(timeout_s)

    def execute(self, circuit: QuantumCircuit, shots: int) -> QuantumBackendResult:
        started = time.perf_counter()
        if QbraidProvider is None:
            raise RuntimeError("qBraid SDK is not installed in the active Python environment")
        if not self.api_key:
            raise RuntimeError("QBRAID_API_KEY is not configured in the environment or .env file")

        provider = QbraidProvider(api_key=self.api_key)
        device = provider.get_device(self.device_id)
        qasm_program = qasm3.dumps(circuit)
        job = device.run(qasm_program, shots=max(int(shots), 1))
        result = job.result(timeout=self.timeout_s)
        counts = _extract_qbraid_counts(result, circuit.num_qubits)

        return QuantumBackendResult(
            counts=counts,
            backend_name=self.name,
            runtime_s=time.perf_counter() - started,
            note=f"Executed final optimized circuit through qBraid device {self.device_id}.",
            job_id=str(getattr(job, "id", "")) or None,
            metadata={"device_id": self.device_id},
        )


def run_qaoa(
    qubo: QuboModel,
    reps: int = DEFAULT_QAOA_REPS,
    shots: int = DEFAULT_SHOTS,
    max_qubits: int = DEFAULT_MAX_QUBITS,
    seed: int | None = None,
    optimizer_maxiter: int = 80,
    execution_iterations: int = 1,
    backend_name: str = "local_aer",
    qbraid_device_id: str | None = None,
    qbraid_timeout_s: int = 300,
) -> dict[str, object]:
    """Optimize QAOA parameters locally, then execute the final circuit repeatedly."""

    started = time.perf_counter()
    if qubo.size > max_qubits:
        return _failed_qaoa_result(
            started,
            "skipped",
            f"QUBO needs {qubo.size} qubits for {qubo.candidate_count} candidates; "
            f"max_qubits is {max_qubits}.",
        )

    iteration_count = max(int(execution_iterations), 1)
    shots_per_iteration = max(int(shots), 1)
    rng = np.random.default_rng(seed)
    local_optimizer_backend = LocalAerBackend(seed=seed)
    linear, quadratic = qubo_to_ising(qubo)

    def expected_energy(params: np.ndarray) -> float:
        circuit = build_qaoa_circuit(qubo.size, linear, quadratic, params, reps, measured=False)
        probabilities = aer_state_probabilities(circuit, local_optimizer_backend)
        return float(sum(prob * qubo.energy(bits) for bits, prob in probabilities))

    params, expectation, optimizer_name = optimize_qaoa_parameters(
        expected_energy,
        reps=max(int(reps), 1),
        maxiter=optimizer_maxiter,
        rng=rng,
    )
    final_circuit = build_qaoa_circuit(
        qubo.size,
        linear,
        quadratic,
        params,
        reps,
        measured=True,
    )
    circuit_diagram = str(final_circuit.draw(output="text"))

    execution_backend_name = backend_name
    combined_counts: dict[str, int] = {}
    backend_runtime_s = 0.0
    execution_notes: list[str] = []
    job_ids: list[str] = []
    try:
        for iteration in range(iteration_count):
            execution_backend = create_quantum_backend(
                backend_name,
                seed=_iteration_seed(seed, iteration),
                qbraid_device_id=qbraid_device_id,
                qbraid_timeout_s=qbraid_timeout_s,
            )
            execution_backend_name = execution_backend.name
            backend_result = execution_backend.execute(final_circuit, shots=shots_per_iteration)
            combined_counts = _merge_counts(combined_counts, backend_result.counts)
            backend_runtime_s += backend_result.runtime_s
            if backend_result.note and backend_result.note not in execution_notes:
                execution_notes.append(backend_result.note)
            if backend_result.job_id:
                job_ids.append(str(backend_result.job_id))
    except Exception as exc:
        return {
            **_failed_qaoa_result(
                started,
                "failed",
                f"{execution_backend_name} execution failed: {exc}",
            ),
            "parameters": params.tolist(),
            "best_energy_expectation": expectation,
            "optimization_backend": local_optimizer_backend.name,
            "execution_backend": execution_backend_name,
            "execution_iterations": iteration_count,
            "shots_per_iteration": shots_per_iteration,
            "total_shots": shots_per_iteration * iteration_count,
            "circuit_diagram": circuit_diagram,
            "circuit_depth": final_circuit.depth(),
            "circuit_width": final_circuit.width(),
            "circuit_num_qubits": final_circuit.num_qubits,
            "circuit_gate_counts": dict(final_circuit.count_ops()),
            "candidate_count": qubo.candidate_count,
            "basis_state_count": qubo.state_count,
            "invalid_state_count": qubo.invalid_state_count,
        }

    samples = _samples_from_counts(combined_counts, qubo)
    return {
        "algorithm": "QAOA",
        "status": "ok",
        "samples": samples,
        "runtime_s": time.perf_counter() - started,
        "backend_runtime_s": backend_runtime_s,
        "best_energy_expectation": expectation,
        "parameters": params.tolist(),
        "optimization_backend": local_optimizer_backend.name,
        "execution_backend": execution_backend_name,
        "execution_iterations": iteration_count,
        "shots_per_iteration": shots_per_iteration,
        "total_shots": shots_per_iteration * iteration_count,
        "measurement_counts": combined_counts,
        "job_id": job_ids[-1] if job_ids else None,
        "job_ids": job_ids,
        "circuit_diagram": circuit_diagram,
        "circuit_depth": final_circuit.depth(),
        "circuit_width": final_circuit.width(),
        "circuit_num_qubits": final_circuit.num_qubits,
        "circuit_gate_counts": dict(final_circuit.count_ops()),
        "candidate_count": qubo.candidate_count,
        "basis_state_count": qubo.state_count,
        "invalid_state_count": qubo.invalid_state_count,
        "note": (
            f"Optimized compact binary-index QAOA gamma/beta locally with Qiskit Aer "
            f"using {optimizer_name}; "
            f"executed final circuit {iteration_count} time(s) with "
            f"{shots_per_iteration} shots each. {' '.join(execution_notes)}"
        ),
    }


def run_qaqo(*args, **kwargs) -> dict[str, object]:
    """Alias kept for the QAQO spelling used in early project notes."""

    return run_qaoa(*args, **kwargs)


def create_quantum_backend(
    backend_name: str,
    seed: int | None = None,
    qbraid_device_id: str | None = None,
    qbraid_timeout_s: int = 300,
) -> QuantumBackend:
    """Create the requested final-circuit execution backend."""

    normalized = backend_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"local", "local_aer", "aer", "qiskit_aer"}:
        return LocalAerBackend(seed=seed)
    if normalized in {"qbraid", "qbraid_final_circuit", "qbraidbackend"}:
        return QBraidBackend(device_id=qbraid_device_id, timeout_s=qbraid_timeout_s)
    raise ValueError(f"Unknown quantum backend: {backend_name}")


def build_qaoa_circuit(
    num_qubits: int,
    linear_terms: np.ndarray,
    quadratic_terms: dict[tuple[int, int], float],
    params: np.ndarray,
    reps: int,
    measured: bool,
) -> QuantumCircuit:
    """Build the numeric QAOA circuit for a given parameter vector."""

    depth = max(int(reps), 1)
    gammas = params[:depth]
    betas = params[depth:]
    circuit = QuantumCircuit(num_qubits, num_qubits if measured else 0)
    circuit.h(range(num_qubits))

    for gamma, beta in zip(gammas, betas):
        _apply_cost_layer(circuit, linear_terms, quadratic_terms, float(gamma))
        _apply_mixer_layer(circuit, num_qubits, float(beta))

    if measured:
        circuit.measure(range(num_qubits), range(num_qubits))

    return circuit


def qubo_to_ising(qubo: QuboModel) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Convert upper-triangular QUBO coefficients to Z and ZZ coefficients."""

    size = qubo.size
    linear = np.zeros(size, dtype=float)
    quadratic: dict[tuple[int, int], float] = {}

    upper = np.triu(qubo.matrix)
    for i in range(size):
        qii = upper[i, i]
        linear[i] += -qii / 2.0
        for j in range(i + 1, size):
            qij = upper[i, j]
            if abs(qij) <= 1e-12:
                continue
            linear[i] += -qij / 4.0
            linear[j] += -qij / 4.0
            quadratic[(i, j)] = quadratic.get((i, j), 0.0) + qij / 4.0

    return linear, quadratic


def aer_state_probabilities(
    circuit: QuantumCircuit,
    backend: LocalAerBackend,
) -> list[tuple[list[int], float]]:
    """Return exact state probabilities from an unmeasured circuit using Aer."""

    state_circuit = circuit.copy()
    state_circuit.save_statevector()
    state_backend = AerSimulator(method="statevector", seed_simulator=backend.seed)
    result = state_backend.run(state_circuit, seed_simulator=backend.seed).result()
    statevector = np.asarray(result.get_statevector(state_circuit), dtype=complex)

    probabilities: list[tuple[list[int], float]] = []
    for basis_index, amplitude in enumerate(statevector):
        probability = float(abs(amplitude) ** 2)
        if probability <= 1e-15:
            continue
        bits = [(basis_index >> bit_index) & 1 for bit_index in range(circuit.num_qubits)]
        probabilities.append((bits, probability))
    return probabilities


def optimize_qaoa_parameters(
    objective: Callable[[np.ndarray], float],
    reps: int,
    maxiter: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, str]:
    """Find gamma and beta values using local classical optimization."""

    depth = max(int(reps), 1)
    initial = np.concatenate(
        [rng.uniform(0, np.pi, size=depth), rng.uniform(0, np.pi / 2, size=depth)]
    )
    minimum_iterations = initial.size + 2
    result = minimize(
        objective,
        initial,
        method="COBYLA",
        options={"maxiter": max(int(maxiter), minimum_iterations), "rhobeg": 0.4},
    )
    return np.asarray(result.x, dtype=float), float(result.fun), "COBYLA"


def _apply_cost_layer(
    circuit: QuantumCircuit,
    linear_terms: np.ndarray,
    quadratic_terms: dict[tuple[int, int], float],
    gamma: float,
) -> None:
    """Apply the Ising cost Hamiltonian phase-separation layer."""

    for index, coefficient in enumerate(linear_terms):
        if abs(coefficient) > 1e-12:
            circuit.rz(2.0 * gamma * coefficient, index)
    for (left, right), coefficient in quadratic_terms.items():
        if abs(coefficient) > 1e-12:
            circuit.rzz(2.0 * gamma * coefficient, left, right)


def _apply_mixer_layer(circuit: QuantumCircuit, num_qubits: int, beta: float) -> None:
    """Apply the standard transverse-field QAOA mixer layer."""

    for index in range(num_qubits):
        circuit.rx(2.0 * beta, index)


def _samples_from_counts(counts: dict[str, int], qubo: QuboModel) -> list[dict[str, object]]:
    """Expand backend measurement counts into benchmark sample dictionaries."""

    samples: list[dict[str, object]] = []
    for bitstring, count in counts.items():
        bits = _bitstring_to_bits(bitstring, qubo.size)
        sample = sample_from_bits(bits, qubo.scores)
        sample["energy"] = qubo.energy(bits)
        samples.extend([sample] * int(count))
    return samples


def _merge_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    """Combine measurement counts from repeated circuit executions."""

    merged = dict(left)
    for bitstring, count in right.items():
        merged[bitstring] = merged.get(bitstring, 0) + int(count)
    return merged


def _iteration_seed(seed: int | None, iteration: int) -> int | None:
    """Use reproducible but distinct simulator seeds for repeated executions."""

    if seed is None:
        return None
    return int(seed) + int(iteration)


def _extract_qbraid_counts(result: object, num_qubits: int) -> dict[str, int]:
    """Extract measurement counts from qBraid result objects."""

    data = getattr(result, "data", None)
    if data is not None and hasattr(data, "get_counts"):
        return _normalize_counts(data.get_counts(), num_qubits)
    if hasattr(result, "get_counts"):
        return _normalize_counts(result.get_counts(), num_qubits)
    if data is not None and hasattr(data, "measurement_counts"):
        return _normalize_counts(data.measurement_counts, num_qubits)
    raise RuntimeError("qBraid result did not include measurement counts")


def _normalize_counts(raw_counts: object, num_qubits: int) -> dict[str, int]:
    """Normalize backend-specific count keys to Qiskit-style binary strings."""

    if isinstance(raw_counts, list):
        if not raw_counts:
            return {}
        raw_counts = raw_counts[0]
    if not isinstance(raw_counts, dict):
        raise RuntimeError("measurement counts are not a dictionary")

    normalized: dict[str, int] = {}
    for raw_key, raw_count in raw_counts.items():
        bitstring = _normalize_bitstring(raw_key, num_qubits)
        normalized[bitstring] = normalized.get(bitstring, 0) + int(raw_count)
    return normalized


def _normalize_bitstring(raw_key: object, num_qubits: int) -> str:
    """Convert decimal or binary backend keys to a fixed-width bitstring."""

    if isinstance(raw_key, int):
        return format(raw_key, f"0{num_qubits}b")

    text = str(raw_key).replace(" ", "").strip()
    if text.startswith("0b"):
        return text[2:].zfill(num_qubits)
    if set(text) <= {"0", "1"}:
        return text.zfill(num_qubits)
    if text.isdigit():
        return format(int(text), f"0{num_qubits}b")
    raise RuntimeError(f"Unsupported measurement key: {raw_key}")


def _bitstring_to_bits(bitstring: str, num_qubits: int) -> list[int]:
    """Convert Qiskit count keys to qubit-index order used by QKash."""

    normalized = _normalize_bitstring(bitstring, num_qubits)
    return [int(bit) for bit in normalized[-num_qubits:][::-1]]


def _failed_qaoa_result(started: float, status: str, note: str) -> dict[str, object]:
    """Create a standard failed/skipped QAOA result."""

    return {
        "algorithm": "QAOA",
        "status": status,
        "samples": [],
        "runtime_s": time.perf_counter() - started,
        "note": note,
    }
