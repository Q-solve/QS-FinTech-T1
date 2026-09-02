# QAOA experiment

This document describes how QKash runs QAOA, how measured bitstrings are decoded, and what the
research dashboard metrics can support.

## Scope

QAOA runs only after `verify_qubo_equivalence` confirms that the QUBO optimum is the exact optimum
of the original constrained candidate model. If equivalence fails, `app.py` prints
`QUBO Validation failed` and does not call `run_qaoa`.

The QUBO uses **one-hot encoding**, so the circuit width is one qubit per candidate row:

```text
qubits = candidate rows
```

`DEFAULT_MAX_QUBITS` (20) caps local and qBraid execution, and `MODEL_CANDIDATE_CAP` in `app.py`
matches it so the candidate set is capped before the QUBO is built. If a run still exceeds the cap,
`run_qaoa` returns `status = "skipped"` with the reason instead of truncating candidates.

See [Why not a compact binary-index encoding](qubo_formulation.md#why-not-a-compact-binary-index-encoding)
for why the qubit count is linear rather than logarithmic in the candidate count.

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

`QuboModel.decode_index(bits)` dispatches on the model's `encoding`. Under one-hot a bit list is
feasible only when exactly one bit is set, and the selected index is that bit's position; every
other bitstring is infeasible. Of the `2**n` basis states only `n` are feasible, so most of the
Hilbert space is infeasible by construction and the penalty term is what suppresses it.

`sample_from_bits` returns the common benchmark sample shape:

```text
bits, index, objective, feasible
```

The `objective` is the decoded candidate's weighted score. It is not the raw QUBO energy. QUBO
energy is retained separately in QAOA samples for debugging.

## Output validation

`validate_solver_outputs` runs before metrics are computed. It checks that every sample:

- has binary bits of length `candidate_count` (one-hot width);
- has a feasibility flag consistent with its actual Hamming weight;
- reports no index for infeasible bitstrings;
- reports the selected candidate index for feasible states;
- reports an objective equal to that candidate's weighted score, within `1e-9`.

A solver reporting an index inconsistent with its own bitstring is rejected, not corrected.

If any solver fails validation, `app.py` stops before showing metrics.

## Metrics

All metrics are defined in `qkash/benchmark.py` and applied identically to every solver.

### Quality

| Metric | Definition |
| --- | --- |
| `feasibility_rate` | Fraction of samples with exactly one bit set (a valid one-hot selection). |
| `objective_value` | Best weighted score among feasible samples. |
| `relative_optimality_gap` | `(best - exact) / max(abs(exact), 1e-12)`, floored at 0. |
| `optimum_hit_probability` | Fraction of **all** samples that are feasible and select a co-optimal index. |
| `stability` | Modal share of the selected index, among **feasible** samples only. |

Note the asymmetry: `optimum_hit_probability` divides by all samples, `stability` only by feasible
ones. A solver that is rarely feasible but consistent when it is will show low hit probability and
high stability. That is informative, not contradictory.

### Cost — three clocks

Comparing a queued remote device against a local simulator on wall clock measures the provider's
queue, not the algorithm. QKash therefore reports three separate clocks.

| Metric | Definition |
| --- | --- |
| `end_to_end_runtime_s` | Total wall clock: local optimization plus, for a remote backend, submission, network, and queue waiting. The honest total cost of using the backend. |
| `compute_runtime_s` | Work the algorithm actually did, with submission/network/queue removed. Defined for every solver — classical solvers do no remote work, so it equals their wall clock. |
| `device_runtime_s` | Circuit execution on the quantum device only. `NaN` for solvers that never touch one. |
| `optimizer_runtime_s` | The local COBYLA loop, broken out (QAOA result dictionary only). |

`time_to_solution_s` and `compute_time_to_solution_s` apply the repeat-until-success estimate
`runtime_per_run * log(1 - confidence) / log(1 - p_hit)` to the first two clocks.

**Ranking uses the compute clock.** `METRIC_RANKING_RULE` orders by gap, objective, hit probability,
feasibility, then `compute_time_to_solution_s` and `compute_runtime_s`. `end_to_end_runtime_s` stays
in the table but does not decide the ordering, so a busy provider queue cannot change which solver
appears best. `METRIC_RANKING_FALLBACKS` maps each compute column to its wall-clock equivalent, so a
metrics frame lacking the compute clock still orders correctly instead of tying every row.

### qBraid device time

`device_runtime_s` for a qBraid job comes from `TimeStamps.executionDuration`, measured in
milliseconds around the simulation itself in `qbraid_core`, excluding queue and network.

**Caveat**: when the device reports nothing, the qBraid schema derives that field from
`endedAt - createdAt`, and `createdAt` is job *creation*, so the fallback **includes queue time**.
`_extract_qbraid_device_runtime` detects this (the two agree within 1 ms) and labels the result
`"qBraid endedAt-createdAt (includes queue)"` rather than passing it off as machine time. Check
`device_time_source` before quoting a device number.

The qBraid SDK exposes **no seed** on the native submission path, so repeated remote executions of a
fixed circuit cannot be varied.

**Unverified**: the extractor is covered by unit tests against mock timestamp objects but has never
run against a live qBraid job. Confirm with one real run before relying on the numbers.

## Reference results

Instance: the five-candidate Kenya to Tanzania QUBO from
[`qubo_formulation.md`](qubo_formulation.md#reference-instance), `A = 2.0`, exact optimum `0.16316`
at candidate 0, unique. Backend `LocalAerBackend` (noiseless), 512 shots, **25 seeds** per depth.

The correct baseline is uniform sampling over all 32 states, which the initial Hadamard layer
produces before any cost or mixer layer:

| Uniform baseline | Value |
| --- | --- |
| Feasibility rate | 5/32 = 0.1562 |
| Optimum hit probability | 1/32 = 0.0312 |
| Optimum hit **given feasible** | 1/5 = 0.2000 |

Measured, mean over 25 seeds with a 95% t-interval on the conditional hit rate:

| `p` | Depth | Feasibility rate | Hit given feasible | 95% CI | vs 0.2000 |
| --- | --- | --- | --- | --- | --- |
| 1 | 11 | 0.3808 | 0.2154 | [0.1986, 0.2322] | p = 0.070, not significant |
| 2 | 18 | 0.5210 | 0.1987 | [0.1768, 0.2206] | p = 0.906, not significant |
| 3 | 25 | 0.5870 | 0.2198 | [0.1871, 0.2524] | p = 0.223, not significant |

Gate counts scale linearly with depth: `p = 1` is 10 `rzz`, 5 `rz`, 5 `rx`, 5 `h`, 5 `measure`.

### What this shows

**QAOA learns the constraint.** Feasibility rises from the uniform 0.1562 to 0.38 at `p = 1` and
0.52 to 0.59 at higher depth, a 2.4x to 3.8x improvement. The one-hot penalty is large and uniform
across candidates, so the cost layer suppresses infeasible states effectively.

**QAOA does not learn the objective.** Conditioned on feasibility, the probability of selecting the
optimal candidate is statistically indistinguishable from uniform at every depth tested. All three
confidence intervals contain 0.2000, and no p-value is below 0.05. Depth does not help.

This is the near-degeneracy predicted in
[Ising conversion](qubo_formulation.md#ising-conversion): all ten couplings are identical at
`J = 1.0`, and the local fields, the only candidate-specific information in the Hamiltonian, span
just 2.5% of their own magnitude. The penalty structure dominates the cost Hamiltonian so completely
that the objective is nearly invisible to it.

**A smaller seed count would have produced a false finding.** At 10 seeds the same experiment gave
conditional hit rates of 0.2208, 0.2046, and 0.2428, which reads as a modest improvement over
uniform. It does not survive 25 seeds and a confidence interval. Report distributions, not point
estimates from a handful of runs.

**The zero optimality gap is an artifact of shot count.** `objective_value` is the best objective
over all samples, so with 512 shots the optimum appears somewhere in essentially every run and
`relative_optimality_gap` is 0.0 throughout. Best-of-512 from a five-element set finds the minimum
whether or not the sampler is informed. `optimum_hit_probability` is the honest metric here, and the
ratio against uniform is the honest comparison.

## Interpretation limits

- **Five qubits is a demonstration, not evidence.** The instance is solved exactly by enumeration in
  microseconds. A QAOA result here validates the implementation; it says nothing about advantage.
- **The local backend is noiseless.** Aer results carry no gate error, decoherence, or readout
  error, so they are an upper bound on hardware.
- **Deterministic solvers return one sample; SA returns `num_reads`; QAOA returns
  `shots * execution_iterations`.** Those are different sampling regimes. Read `stability` per
  solver rather than ranking on it.
- **Never present QAOA output as a commercial recommendation.** It selects a row from a candidate
  set whose limitations are documented in
  [`classical_model.md`](classical_model.md#limitations).

A meaningful advantage experiment needs the harder research mode: many transfers at once, hard
cost/time/risk eligibility constraints, provider concentration limits, repeated seeds, and the
runtime accounting above.

## Reproducing

```bash
python - <<PY
import numpy as np
from scipy import stats
from qkash.scoring import build_selection_qubo, solve_original_exact
from qkash.quantum import run_qaoa
from qkash.benchmark import summarize_samples

scores = [0.1631558700780258, 0.1763382326256578, 0.2,
          0.25926739248775177, 0.32]
qubo = build_selection_qubo(scores, penalty=2.0)
exact = solve_original_exact(scores)

conditional = []
for seed in range(25):
    r = run_qaoa(qubo, reps=1, shots=512, max_qubits=20, seed=seed,
                 backend_name="local_aer")
    m = summarize_samples("QAOA", r["samples"], float(exact["objective"]),
                          exact["indices"], float(r["runtime_s"]))
    if m["feasibility_rate"] > 0:
        conditional.append(m["optimum_hit_probability"] / m["feasibility_rate"])

arr = np.array(conditional)
print("hit|feasible:", arr.mean())
print("95% CI:", stats.t.interval(0.95, len(arr) - 1, loc=arr.mean(),
                                  scale=stats.sem(arr)))
print("vs uniform 0.20, p =", stats.ttest_1samp(arr, 0.20).pvalue)
PY
```

Record the seed list with every result.
