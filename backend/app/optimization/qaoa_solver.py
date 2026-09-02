"""Local, finite-shot QAOA execution for an already validated QUBO model."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, pi
from time import perf_counter
from typing import Any

import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

from .classical_solver import is_exactly_one
from .objective import SCORE_TIE_TOLERANCE
from .qaoa_models import (
    CircuitMetrics,
    QaoaCallbackRecord,
    QaoaConfig,
    QaoaRunMetrics,
    QaoaRunResult,
    SampledState,
    validate_probability,
)
from .qubo import QuboModel, qubo_energy
from .qubo_solver import to_ising


class _TimedJob:
    """Delegate a primitive job while measuring its blocking result call."""

    def __init__(self, job: Any, timings: list[float]) -> None:
        self._job = job
        self._timings = timings

    def result(self) -> Any:
        started = perf_counter()
        result = self._job.result()
        self._timings.append(perf_counter() - started)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._job, name)


class _TimedSamplerV2(SamplerV2):
    """Aer sampler that records each completed local primitive job."""

    def __init__(self, *, default_shots: int, seed: int) -> None:
        super().__init__(default_shots=default_shots, seed=seed)
        self.job_result_times: list[float] = []

    def run(self, pubs: Any, *, shots: int | None = None) -> _TimedJob:
        return _TimedJob(super().run(pubs, shots=shots), self.job_result_times)


def qiskit_label_to_x_order(label: str, qubit_count: int) -> tuple[int, ...]:
    """Convert Qiskit's x_(n-1)..x_0 display label to x_0..x_(n-1)."""

    compact = label.replace(" ", "")
    if len(compact) != qubit_count or set(compact) - {"0", "1"}:
        raise ValueError(f"Expected a {qubit_count}-bit Qiskit state label.")
    return tuple(int(bit) for bit in reversed(compact))


def x_order_to_qiskit_label(bits: tuple[int, ...]) -> str:
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise ValueError("Expected a nonempty sequence of binary values.")
    return "".join(str(bit) for bit in reversed(bits))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imaginary": value.imag}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def exact_feasible_optimal_states(
    model: QuboModel,
) -> tuple[tuple[tuple[int, ...], ...], float]:
    """Return every exactly-one state tied at the classical optimum."""

    best_score = min(model.representation.scores)
    states = tuple(
        tuple(
            1 if position == index else 0 for position in range(len(model.alternatives))
        )
        for index, score in enumerate(model.representation.scores)
        if isclose(score, best_score, rel_tol=0.0, abs_tol=SCORE_TIE_TOLERANCE)
    )
    return states, best_score


def _sampled_state(model: QuboModel, label: str, probability: float) -> SampledState:
    bits = qiskit_label_to_x_order(label, len(model.alternatives))
    feasible = is_exactly_one(bits)
    index = bits.index(1) if feasible else None
    alternative = model.alternatives[index] if index is not None else None
    return SampledState(
        qiskit_state_label_xn_to_x0=label.replace(" ", ""),
        bit_string_x0_to_xn="".join(str(bit) for bit in bits),
        probability=validate_probability(probability, "sample probability"),
        feasible=feasible,
        qubo_energy=qubo_energy(model, bits),
        weighted_score=(
            model.representation.scores[index] if index is not None else None
        ),
        selected_alternative_id=(alternative.alternative_id if alternative else None),
        provider=(alternative.firm if alternative else None),
        payment_instrument=(alternative.payment_instrument if alternative else None),
        pickup_method=(alternative.pickup_method if alternative else None),
    )


def interpret_sample_distribution(
    model: QuboModel,
    probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Interpret raw samples without repairing or replacing their modal state."""

    if not probabilities:
        raise ValueError("QAOA returned an empty final sample distribution.")
    total_probability = float(sum(probabilities.values()))
    if not isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"Final sample probabilities must sum to 1; got {total_probability}."
        )

    sampled = {
        label.replace(" ", ""): _sampled_state(model, label, probability)
        for label, probability in probabilities.items()
    }
    raw = min(
        sampled.values(),
        key=lambda state: (-state.probability, state.qiskit_state_label_xn_to_x0),
    )
    feasible_sampled = [state for state in sampled.values() if state.feasible]
    rank_by_id = {
        item.alternative.alternative_id: item.rank for item in model.ranked_alternatives
    }
    best_feasible = (
        min(
            feasible_sampled,
            key=lambda state: (
                state.weighted_score,
                rank_by_id[state.selected_alternative_id],
            ),
        )
        if feasible_sampled
        else None
    )

    exact_states, exact_score = exact_feasible_optimal_states(model)
    exact_labels = tuple(x_order_to_qiskit_label(bits) for bits in exact_states)
    feasible_labels = tuple(
        x_order_to_qiskit_label(
            tuple(
                1 if position == index else 0
                for position in range(len(model.alternatives))
            )
        )
        for index in range(len(model.alternatives))
    )
    feasible_probabilities = {
        label: sampled[label].probability if label in sampled else 0.0
        for label in feasible_labels
    }
    feasible_probability = validate_probability(
        sum(feasible_probabilities.values()), "feasible probability"
    )
    optimal_probability = validate_probability(
        sum(sampled[label].probability for label in exact_labels if label in sampled),
        "optimal-state probability",
    )
    exact_is_most_probable = any(
        label in sampled
        and isclose(
            sampled[label].probability,
            raw.probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for label in exact_labels
    )
    best_score = best_feasible.weighted_score if best_feasible else None
    gap = abs(best_score - exact_score) if best_score is not None else None
    return {
        "raw": raw,
        "best_feasible": best_feasible,
        "feasible_state_probabilities": feasible_probabilities,
        "feasible_probability": feasible_probability,
        "exact_states": exact_states,
        "exact_labels": exact_labels,
        "exact_score": exact_score,
        "optimal_probability": optimal_probability,
        "exact_sampled": optimal_probability > 0.0,
        "exact_most_probable": exact_is_most_probable,
        "best_score": best_score,
        "gap": gap,
    }


def _deterministic_initial_point(ansatz: QAOAAnsatz, seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    values = []
    for lower, upper in ansatz.parameter_bounds:
        lower_bound = -2.0 * pi if lower is None else lower
        upper_bound = 2.0 * pi if upper is None else upper
        values.append(random.uniform(lower_bound, upper_bound))
    return np.asarray(values, dtype=float)


def run_qaoa(model: QuboModel, config: QaoaConfig) -> QaoaRunResult:
    """Run standard-X-mixer QAOA against ``model.qubo_problem.to_ising()``."""

    total_started = perf_counter()
    ising_started = perf_counter()
    operator, ising_offset = to_ising(model)
    ising_elapsed = perf_counter() - ising_started
    circuit_started = perf_counter()
    original_ansatz = QAOAAnsatz(operator, reps=config.reps, flatten=True)
    original_depth = original_ansatz.depth()
    decomposed_depth = original_ansatz.decompose(reps=10).depth()
    circuit_elapsed = perf_counter() - circuit_started
    backend = AerSimulator()
    pass_manager = generate_preset_pass_manager(
        optimization_level=config.transpiler_optimization_level,
        backend=backend,
        seed_transpiler=config.resolved_transpiler_seed,
    )
    transpilation_started = perf_counter()
    transpiled_ansatz = pass_manager.run(original_ansatz)
    transpilation_elapsed = perf_counter() - transpilation_started
    circuit_metrics = CircuitMetrics(
        original_depth=original_depth,
        decomposed_depth=decomposed_depth,
        transpiled_depth=transpiled_ansatz.depth(),
        number_of_parameters=original_ansatz.num_parameters,
    )

    callback_history: list[QaoaCallbackRecord] = []

    def callback(
        evaluation_number: int,
        parameters: np.ndarray,
        expectation_value: float,
        metadata: dict[str, Any],
    ) -> None:
        callback_history.append(
            QaoaCallbackRecord(
                evaluation_number=int(evaluation_number),
                parameters=tuple(float(value) for value in parameters),
                expectation_value=float(expectation_value),
                metadata=_json_value(metadata),
            )
        )

    sampler = _TimedSamplerV2(
        default_shots=config.shots,
        seed=config.resolved_simulator_seed,
    )
    qaoa = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=config.max_iterations),
        reps=config.reps,
        initial_point=_deterministic_initial_point(
            original_ansatz, config.algorithm_seed
        ),
        callback=callback,
        transpiler=pass_manager,
    )
    result = qaoa.compute_minimum_eigenvalue(operator)
    total_elapsed = perf_counter() - total_started
    if not sampler.job_result_times:
        raise RuntimeError("QAOA completed without a recorded sampler job.")
    optimizer_sampler_time = sum(sampler.job_result_times[:-1])
    final_sampling_time = sampler.job_result_times[-1]
    probabilities = {
        str(label): float(value) for label, value in result.eigenstate.items()
    }
    interpreted = interpret_sample_distribution(model, probabilities)
    best_feasible = interpreted["best_feasible"]
    success = best_feasible is not None
    message = (
        "At least one feasible exactly-one state was sampled."
        if success
        else "No feasible exactly-one state was sampled; no recommendation is returned."
    )
    metrics = QaoaRunMetrics(
        qubit_count=operator.num_qubits,
        qaoa_repetitions=config.reps,
        shots=config.shots,
        algorithm_seed=config.algorithm_seed,
        simulator_seed=config.resolved_simulator_seed,
        transpiler_seed=config.resolved_transpiler_seed,
        optimizer=config.optimizer,
        maximum_iterations=config.max_iterations,
        actual_evaluations=int(result.cost_function_evals),
        ising_conversion_time_seconds=ising_elapsed,
        circuit_construction_time_seconds=circuit_elapsed,
        transpilation_time_seconds=transpilation_elapsed,
        optimization_time_seconds=float(result.optimizer_time),
        optimizer_sampler_time_seconds=optimizer_sampler_time,
        final_sampling_time_seconds=final_sampling_time,
        queue_time_seconds=0.0,
        total_execution_time_seconds=total_elapsed,
        circuit=circuit_metrics,
    )
    return QaoaRunResult(
        success=success,
        message=message,
        raw_most_probable_state=interpreted["raw"],
        best_feasible_sampled_state=best_feasible,
        feasible_state_probabilities=interpreted["feasible_state_probabilities"],
        feasible_probability=interpreted["feasible_probability"],
        exact_optimal_bit_strings_x0_to_xn=tuple(
            "".join(str(bit) for bit in bits) for bits in interpreted["exact_states"]
        ),
        exact_optimal_qiskit_labels_xn_to_x0=interpreted["exact_labels"],
        exact_weighted_score=interpreted["exact_score"],
        optimal_state_probability=interpreted["optimal_probability"],
        exact_optimum_sampled=interpreted["exact_sampled"],
        exact_optimum_most_probable=interpreted["exact_most_probable"],
        best_feasible_weighted_score=interpreted["best_score"],
        absolute_optimality_gap=interpreted["gap"],
        metrics=metrics,
        optimal_parameters=tuple(float(value) for value in result.optimal_point),
        final_expectation_value_without_ising_offset=float(result.eigenvalue),
        ising_offset=float(ising_offset),
        callback_history=tuple(callback_history),
    )
