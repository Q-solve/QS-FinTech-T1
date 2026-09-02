"""Mathematical-optimization baseline for the constrained provider model."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from qiskit_optimization.algorithms import ScipyMilpOptimizer

from .classical_solver import is_exactly_one
from .models import ServiceAlternative
from .qubo import QuboModel


@dataclass(frozen=True)
class MathematicalOptimizationResult:
    solver_name: str
    bit_values: tuple[int, ...]
    feasible: bool
    weighted_score: float
    selected_alternative: ServiceAlternative | None
    runtime_seconds: float
    success: bool
    status: str
    message: str
    mip_gap: float | None
    mip_dual_bound: float | None
    mip_node_count: int | None


def solve_constrained_with_scipy_milp(
    model: QuboModel,
) -> MathematicalOptimizationResult:
    """Solve the pre-penalty exactly-one program through SciPy/HiGHS MILP."""

    started = perf_counter()
    result = ScipyMilpOptimizer().solve(model.constrained_problem)
    runtime = perf_counter() - started
    bits = tuple(round(value) for value in result.x)
    feasible = is_exactly_one(bits)
    selected_index = bits.index(1) if feasible else None
    raw = result.raw_results
    return MathematicalOptimizationResult(
        solver_name="qiskit_scipy_milp_highs",
        bit_values=bits,
        feasible=feasible,
        weighted_score=float(result.fval),
        selected_alternative=(
            model.alternatives[selected_index] if selected_index is not None else None
        ),
        runtime_seconds=runtime,
        success=bool(raw.success),
        status=result.status.name,
        message=str(raw.message),
        mip_gap=float(raw.mip_gap) if raw.mip_gap is not None else None,
        mip_dual_bound=(
            float(raw.mip_dual_bound) if raw.mip_dual_bound is not None else None
        ),
        mip_node_count=(
            int(raw.mip_node_count) if raw.mip_node_count is not None else None
        ),
    )
