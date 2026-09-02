"""Classical optimization baselines for QKash."""

from __future__ import annotations

# Imports.
import math
import time

import numpy as np
import pandas as pd

from .benchmark import deterministic_sample, sample_from_bits
from .scoring import QuboModel, solve_original_exact


def business_as_usual_baseline(candidates: pd.DataFrame) -> dict[str, object]:
    """Select the most commonly observed firm after filtering.

    This is the non-optimization baseline: it represents continuing with the
    incumbent/most-visible provider in the filtered market, then taking that
    provider's best currently scored row.
    """

    started = time.perf_counter()
    if candidates.empty:
        raise ValueError("No candidates supplied")

    counts = candidates["firm"].value_counts()
    ranked = candidates.copy()
    ranked["_market_presence"] = ranked["firm"].map(counts)
    ranked = ranked.sort_values(
        ["_market_presence", "date_parsed", "weighted_score"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    index = int(ranked.iloc[0]["candidate_index"])
    samples = [deterministic_sample(index, candidates["weighted_score"])]
    return {
        "algorithm": "Business as usual",
        "best_index": index,
        "samples": samples,
        "runtime_s": time.perf_counter() - started,
        "note": "Most frequently observed firm in the filtered dataset.",
    }


def exact_mathematical_baseline(candidates: pd.DataFrame) -> dict[str, object]:
    """Solve the original constrained one-hot model exactly."""

    started = time.perf_counter()
    exact = solve_original_exact(candidates["weighted_score"])
    index = int(exact["index"])
    samples = [deterministic_sample(index, candidates["weighted_score"])]
    return {
        "algorithm": "Exact mathematical baseline",
        "best_index": index,
        "samples": samples,
        "runtime_s": time.perf_counter() - started,
        "note": "Global optimum of min weighted_score subject to selecting one provider.",
    }


def simulated_annealing(
    qubo: QuboModel,
    num_reads: int = 128,
    sweeps: int = 600,
    seed: int | None = None,
    initial_temp: float | None = None,
    final_temp: float = 1e-3,
) -> dict[str, object]:
    """Run a small Metropolis simulated annealer over the QUBO."""

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    size = qubo.size
    reads = max(int(num_reads), 1)
    sweep_count = max(int(sweeps), 1)
    start_temp = float(initial_temp or max(1.0, qubo.penalty + float(np.max(qubo.scores))))
    end_temp = max(float(final_temp), 1e-9)

    samples: list[dict[str, object]] = []
    best_bits: list[int] | None = None
    best_energy = math.inf

    for _ in range(reads):
        bits = rng.integers(0, 2, size=size).astype(int)
        energy = qubo.energy(bits)

        for sweep in range(sweep_count):
            fraction = sweep / max(sweep_count - 1, 1)
            temp = start_temp * ((end_temp / start_temp) ** fraction)
            flip_index = int(rng.integers(0, size))
            proposal = bits.copy()
            proposal[flip_index] = 1 - proposal[flip_index]
            proposal_energy = qubo.energy(proposal)
            delta = proposal_energy - energy

            if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-12)):
                bits = proposal
                energy = proposal_energy

        sample = sample_from_bits(bits, qubo.scores)
        sample["energy"] = float(energy)
        samples.append(sample)
        if energy < best_energy:
            best_energy = float(energy)
            best_bits = bits.astype(int).tolist()

    best_sample = sample_from_bits(best_bits or [0] * size, qubo.scores)
    return {
        "algorithm": "Simulated annealing",
        "best_index": best_sample["index"],
        "samples": samples,
        "runtime_s": time.perf_counter() - started,
        "best_energy": best_energy,
        "note": f"{reads} reads, {sweep_count} sweeps per read.",
    }
