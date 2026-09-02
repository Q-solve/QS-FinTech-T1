"""Typed configuration and result contracts for local QAOA experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from numbers import Integral
from typing import Any


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class QaoaMixer(str, Enum):
    STANDARD_X = "standard_x"
    CONSTRAINT_PRESERVING_XY = "constraint_preserving_xy"


@dataclass(frozen=True)
class QaoaConfig:
    """All stochastic and optimizer settings for one QAOA run."""

    reps: int = 1
    shots: int = 2_048
    max_iterations: int = 60
    algorithm_seed: int = 11
    simulator_seed: int | None = None
    transpiler_seed: int | None = None
    optimizer: str = "COBYLA"
    transpiler_optimization_level: int = 1
    mixer: QaoaMixer = QaoaMixer.STANDARD_X

    def __post_init__(self) -> None:
        object.__setattr__(self, "reps", _positive_integer(self.reps, "reps"))
        object.__setattr__(self, "shots", _positive_integer(self.shots, "shots"))
        object.__setattr__(
            self,
            "max_iterations",
            _positive_integer(self.max_iterations, "max_iterations"),
        )
        for name in ("algorithm_seed", "simulator_seed", "transpiler_seed"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None.")
            if value is not None:
                object.__setattr__(self, name, int(value))
        if self.optimizer != "COBYLA":
            raise ValueError("Only the COBYLA optimizer is currently supported.")
        if self.transpiler_optimization_level not in (0, 1, 2, 3):
            raise ValueError("transpiler_optimization_level must be 0, 1, 2, or 3.")
        try:
            object.__setattr__(self, "mixer", QaoaMixer(self.mixer))
        except ValueError as exc:
            choices = ", ".join(item.value for item in QaoaMixer)
            raise ValueError(f"mixer must be one of: {choices}.") from exc

    @property
    def resolved_simulator_seed(self) -> int:
        return (
            self.algorithm_seed if self.simulator_seed is None else self.simulator_seed
        )

    @property
    def resolved_transpiler_seed(self) -> int:
        return (
            self.algorithm_seed
            if self.transpiler_seed is None
            else self.transpiler_seed
        )


@dataclass(frozen=True)
class QaoaCallbackRecord:
    evaluation_number: int
    parameters: tuple[float, ...]
    expectation_value: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CircuitMetrics:
    original_depth: int
    decomposed_depth: int
    transpiled_depth: int
    number_of_parameters: int


@dataclass(frozen=True)
class SampledState:
    """A measured state with both documented bit-order conventions."""

    qiskit_state_label_xn_to_x0: str
    bit_string_x0_to_xn: str
    probability: float
    feasible: bool
    qubo_energy: float
    weighted_score: float | None
    selected_alternative_id: str | None
    provider: str | None
    payment_instrument: str | None
    pickup_method: str | None


@dataclass(frozen=True)
class QaoaRunMetrics:
    qubit_count: int
    qaoa_repetitions: int
    shots: int
    algorithm_seed: int
    simulator_seed: int
    transpiler_seed: int
    optimizer: str
    maximum_iterations: int
    actual_evaluations: int
    ising_conversion_time_seconds: float
    circuit_construction_time_seconds: float
    transpilation_time_seconds: float
    optimization_time_seconds: float
    optimizer_sampler_time_seconds: float
    final_sampling_time_seconds: float
    queue_time_seconds: float
    total_execution_time_seconds: float
    circuit: CircuitMetrics
    mixer_type: str
    initial_state_type: str
    mixer_connectivity: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class QaoaRunResult:
    success: bool
    message: str
    raw_most_probable_state: SampledState
    best_feasible_sampled_state: SampledState | None
    feasible_state_probabilities: dict[str, float]
    feasible_probability: float
    exact_optimal_bit_strings_x0_to_xn: tuple[str, ...]
    exact_optimal_qiskit_labels_xn_to_x0: tuple[str, ...]
    exact_weighted_score: float
    optimal_state_probability: float
    exact_optimum_sampled: bool
    exact_optimum_most_probable: bool
    best_feasible_weighted_score: float | None
    absolute_optimality_gap: float | None
    metrics: QaoaRunMetrics
    optimal_parameters: tuple[float, ...]
    final_expectation_value_without_ising_offset: float
    ising_offset: float
    callback_history: tuple[QaoaCallbackRecord, ...]
    hamming_weight_distribution: dict[str, float]
    constraint_leakage_probability: float


@dataclass(frozen=True)
class DistributionSummary:
    mean: float | None
    standard_deviation: float | None


@dataclass(frozen=True)
class QaoaAggregate:
    run_count: int
    successful_feasible_run_count: int
    modal_state_feasibility_count: int
    modal_state_feasibility_rate: float
    exact_optimum_recovery_count: int
    exact_optimum_recovery_rate: float
    optimal_state_probability: DistributionSummary
    feasible_probability: DistributionSummary
    best_feasible_gap: DistributionSummary
    runtime_seconds: DistributionSummary
    constraint_leakage_probability: DistributionSummary


@dataclass(frozen=True)
class QaoaExperimentResult:
    schema_version: str
    generated_at_utc: str
    package_versions: dict[str, str]
    experiment_configuration: dict[str, Any]
    exact_reference: dict[str, Any]
    runs: tuple[QaoaRunResult, ...]
    aggregate: QaoaAggregate

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready nested dictionary."""

        return asdict(self)


def validate_probability(value: float, name: str) -> float:
    """Validate public probability helpers without masking numerical errors."""

    number = float(value)
    if not isfinite(number) or number < -1e-12 or number > 1.0 + 1e-12:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")
    return min(1.0, max(0.0, number))
