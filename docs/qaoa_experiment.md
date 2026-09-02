# QAOA experiment

This document describes how QKash runs QAOA, how measured bitstrings are decoded, and what the
research dashboard metrics can support.

## Scope

QAOA runs only after `verify_qubo_equivalence` confirms that the compact QUBO optimum decodes to the
exact optimum of the original constrained candidate model. If equivalence fails, `app.py` prints
`QUBO Validation failed` and does not call `run_qaoa`.

The circuit width is the compact binary-index width:

```text
qubits = ceil(log2(candidate rows))
```

`DEFAULT_MAX_QUBITS` caps local and qBraid execution. If a run exceeds the cap, `run_qaoa` returns
`status = "skipped"` with the reason instead of truncating candidates.

## Algorithm

QKash uses standard gate-model QAOA with a transverse-field mixer.

1. Hadamards initialize a uniform superposition over all basis states.
2. The cost layer applies `rz(2 * gamma * h_i)` and `rzz(2 * gamma * J_ij)` terms from
   `qubo_to_ising`.
3. The mixer layer applies `rx(2 * beta)` on every qubit.
4. Steps 2 and 3 repeat `reps` times.
5. Measurements are added only to the final optimized circuit.

The parameter vector is `[gamma_1..gamma_p, beta_1..beta_p]`.

## Execution order

QAOA always follows this order:

1. Optimize gamma and beta locally with Qiskit Aer and SciPy COBYLA.
2. Construct one measured circuit at the optimized parameters.
3. Execute that final circuit for the configured number of circuit iterations on either
   `LocalAerBackend` or `QBraidBackend`.

qBraid is never used during parameter optimization. Only the final optimized circuit can leave the
machine, and only when the qBraid backend is selected.

## Backends

Both backends implement `QuantumBackend` and return `QuantumBackendResult`.

| Backend | Class | Behaviour |
| --- | --- | --- |
| Local Aer simulator | `LocalAerBackend` | Runs the final optimized measured circuit with Qiskit Aer. |
| qBraid simulator | `QBraidBackend` | Serializes the final circuit to OpenQASM 3 and submits it to the configured qBraid device. |

Remote backend exceptions become a QAOA result with `status = "failed"`, the error message, the
optimized parameters, and circuit metadata. The UI then shows the failure instead of hiding QAOA.

## Bit order and sample interpretation

Qiskit count keys place qubit 0 at the rightmost bit. `_bitstring_to_bits` reverses the count key so
the returned list is in qubit-index order.

`decode_candidate_index(bits, candidate_count)` converts that little-endian bit list to a candidate
row index. Values below `candidate_count` are feasible. Values at or above `candidate_count` are
unused compact-encoding states and are infeasible.

`sample_from_bits` returns the common benchmark sample shape:

```text
bits, index, objective, feasible
```

The `objective` is the decoded candidate's weighted score. It is not the raw QUBO energy. QUBO
energy is retained separately in QAOA samples for debugging.

## Output validation

`validate_solver_outputs` runs before metrics are computed. It checks that every sample:

- has binary bits of length `ceil(log2(candidate_count))`;
- has a feasibility flag consistent with binary-index decoding;
- reports no index for unused basis states;
- reports the decoded candidate index for feasible states;
- reports an objective equal to the candidate weighted score.

If any solver fails validation, `app.py` stops before showing metrics.

## Metrics

All seven research metrics are defined in `qkash/benchmark.py` and applied to every solver:

| Metric | Definition |
| --- | --- |
| `feasibility_rate` | Fraction of samples that decode to real candidate rows. |
| `objective_value` | Best weighted score among feasible samples. |
| `relative_optimality_gap` | `(best - exact) / max(abs(exact), 1e-12)`, floored at 0. |
| `optimum_hit_probability` | Fraction of all samples that decode to an exact-optimal candidate. |
| `end_to_end_runtime_s` | Wall clock for the full solver call. |
| `stability` | Modal share among feasible decoded candidate indices. |
| `time_to_solution_s` | Runtime estimate to hit an optimum at 99% confidence. |

Deterministic solvers return one sample. Simulated annealing returns many reads. QAOA returns
`shots * execution_iterations` samples. Those are different sampling regimes, so rankings should be
read as diagnostics rather than proof of quantum advantage.

## Interpretation limits

The compact QUBO uses fewer qubits than one-hot encoding, but its Hamiltonian is
optimum-preserving. It is suitable for validating the QAOA implementation, backend execution,
decoding, and metric pipeline. It does not by itself demonstrate that QAOA discovers an unknown
commercial optimum better than classical solvers.

A meaningful quantum-advantage experiment should use the harder research mode: many transfers at
once, hard cost/time/risk eligibility constraints, provider concentration limits, repeated seeds,
and transparent runtime accounting.
