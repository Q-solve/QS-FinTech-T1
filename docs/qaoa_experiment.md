# QAOA experiment

This document specifies how QKash runs QAOA, how measurement results are interpreted, which metrics
are reported, and what those metrics can and cannot support. It describes `qkash/quantum.py` and
`qkash/benchmark.py` as implemented.

Reference results for the five-qubit instance are recorded in
[Reference results](#reference-results). They were measured, not estimated, on the local Aer
simulator across 10 seeds at each of `p = 1, 2, 3`. Any new result added to this document must
carry its seed list and backend.

## Scope and source model

QAOA solves the QUBO specified in [`qubo_formulation.md`](qubo_formulation.md) — the same candidate
set, scores, and one-hot constraint as every classical baseline. `verify_qubo_equivalence` must
return `equivalent = True` before `run_qaoa` is called; `app.py` enforces this.

`TARGET_QUBITS = 5` fixes the problem at five qubits. `run_qaoa` refuses anything larger than its
`max_qubits` argument, returning `status = "skipped"` with the reason rather than silently
truncating.

## Algorithm

Standard QAOA with a transverse-field mixer.

1. **Initial state.** Hadamard on every qubit — the uniform superposition over all `2^n` states.
2. **Cost layer.** For each Ising term, `rz(2 * gamma * h_i)` on qubit `i` and
   `rzz(2 * gamma * J_ij)` on each coupled pair. Coefficients at or below `1e-12` are skipped.
3. **Mixer layer.** `rx(2 * beta)` on every qubit.
4. **Repetition.** Layers 2 and 3 repeat `reps` times (`p`, UI range 1–3, default 1).
5. **Measurement.** Added only to the final circuit, never during optimization.

The parameter vector is `[gamma_1..gamma_p, beta_1..beta_p]`, length `2p`.

## Execution order

The order is fixed and is the central design constraint of this module:

1. **Optimize locally.** `gamma` and `beta` are found with Qiskit Aer using SciPy COBYLA. The
   objective is the exact expectation `sum_states P(state) * energy(state)`, computed from the
   statevector of an **unmeasured** circuit — not from sampled shots. This is noiseless and
   deterministic given a seed.
2. **Build the final circuit.** One measured circuit at the optimized parameters.
3. **Execute once.** That single circuit runs on the selected backend.

**No qBraid job is ever submitted during parameter optimization.** COBYLA needs one circuit
evaluation per iteration, up to `optimizer_maxiter` (default 80); routing those through a remote
queue would be slow and expensive for no benefit. Only step 3 leaves the machine.

COBYLA is initialized from `rng.uniform(0, pi)` for the gammas and `rng.uniform(0, pi/2)` for the
betas, with `rhobeg = 0.4`. `maxiter` is floored at `2p + 2`. The seed controls the initial point,
so QAOA output is reproducible end to end on the local backend.

## Backends

Both implement the `QuantumBackend` abstract base class and return a `QuantumBackendResult`.

| Backend | Class | Behaviour |
| --- | --- | --- |
| Local Aer simulator | `LocalAerBackend` | `AerSimulator` with `seed_simulator`. Also used for the optimization loop regardless of the execution backend. |
| qBraid simulator | `QBraidBackend` | Serializes the circuit to OpenQASM 3, submits to a qBraid device, polls to a timeout (default 300 s). |

`QBraidBackend` fails fast and explicitly: it raises if the qBraid SDK is not installed, and raises
if `QBRAID_API_KEY` is absent. `create_quantum_backend` accepts several spellings for each backend
name and raises `ValueError` on anything unrecognized.

A backend exception is caught in `run_qaoa` and returned as `status = "failed"` with the exception
message, the optimized parameters, and the full circuit metadata. A remote failure therefore never
loses the local work and never silently degrades to a simulator result.

## Bit order and sample interpretation

This is the part most likely to be misread, and it is worth stating precisely.

Qiskit count keys are little-endian: the **rightmost** character is qubit 0. `_bitstring_to_bits`
reverses the string so the returned list is in qubit-index order, where element `i` is qubit `i` and
therefore candidate `i`.

`_normalize_counts` and `_normalize_bitstring` accept the several key formats different backends
return — integers, `0b`-prefixed strings, bare binary, and decimal-as-string — and normalize all of
them to fixed-width binary. Keys that collide after normalization have their counts summed. An
unrecognized key raises rather than being dropped.

Each measured bitstring becomes one sample via `sample_from_bits`:

- `one_hot_index` returns the selected index when exactly one bit is set, else `None`.
- `feasible` is `index is not None`.
- `objective` is `scores[index]` when feasible, else `None`.
- `energy` is `QuboModel.energy(bits)`, added by `_samples_from_counts`.

A count of `k` for one bitstring expands to `k` identical samples, so shot-weighted statistics are
correct without re-weighting.

## Output validation

`validate_solver_outputs` runs on every solver's output before metrics are computed, and `app.py`
aborts and shows the validation table if any solver fails. Per sample it checks:

- `bits` is a list of the right length containing only 0 and 1.
- `feasible` agrees with the actual Hamming weight.
- `index` is absent for infeasible bits and present and correct for feasible ones.
- `objective` matches `scores[index]` within `1e-9`.

A solver that reports an index inconsistent with its own bitstring is rejected, not corrected.

## Metrics

All seven are defined in `qkash/benchmark.py` and computed identically for all four solvers.

| Metric | Definition |
| --- | --- |
| `feasibility_rate` | Fraction of samples with exactly one bit set. |
| `objective_value` | Best objective among feasible samples. |
| `relative_optimality_gap` | `(best - exact) / max(abs(exact), 1e-12)`, floored at 0. |
| `optimum_hit_probability` | Fraction of **all** samples that are feasible and select a co-optimal index. |
| `end_to_end_runtime_s` | Wall clock for the whole solver call. |
| `stability` | Modal share — how often the most-frequent selected index was chosen, among feasible samples. |
| `time_to_solution_s` | Runtime to reach the optimum at 99% confidence. |

`time_to_solution` is `runtime_per_run * log(1 - confidence) / log(1 - p_hit)`, the standard
repeat-until-success estimate. It returns `runtime_per_run` when `p_hit = 1`, and infinity when
`p_hit = 0` or no samples exist.

Note the asymmetry between `optimum_hit_probability` and `stability`: the first divides by all
samples, the second only by feasible ones. A solver that is rarely feasible but always picks the
same candidate when it is will show low hit probability and high stability. That combination is
informative, not contradictory.

`rank_metrics` orders solvers by `METRIC_RANKING_RULE` — gap, objective, hit probability,
feasibility, time to solution, runtime, stability — quality first, execution cost as tie-breaker.

## Comparison caveats

The four solvers are **not** measured on equal footing, and any comparison must say so:

- Business as usual and the exact baseline are deterministic and produce exactly **one** sample
  each. Their feasibility rate and stability are 1.0 by construction, and their hit probability is 0
  or 1. These are not distributions.
- Simulated annealing produces `num_reads` samples (default 128).
- QAOA produces `shots` samples (default 512).

Reading `stability` across those four rows compares a single deterministic pick against a 512-shot
distribution. Report it per solver; do not rank on it.

`end_to_end_runtime_s` for QAOA includes the full COBYLA optimization loop — up to 80 statevector
simulations — while the classical baselines include only their own work. On the qBraid backend it
also includes queue and network time. `backend_runtime_s` isolates the final execution, and the
result dictionary carries it separately for exactly this reason.

## Reference results

Instance: the five-candidate Kenya → Tanzania QUBO from
[`qubo_formulation.md`](qubo_formulation.md#reference-instance), `A = 2.0`, exact optimum
`0.059218477950031836` at candidate 0, unique. Backend `LocalAerBackend` (noiseless), 512 shots,
seeds 0–9 at each depth.

The correct baseline is uniform random sampling over all 32 states, which the initial Hadamard layer
produces before any cost or mixer layer is applied:

| Uniform baseline | Value |
| --- | --- |
| Feasibility rate | 5/32 = 0.1562 |
| Optimum hit probability | 1/32 = 0.0312 |
| Optimum hit given feasible | 1/5 = 0.2000 |

Measured, mean across 10 seeds with the observed range in brackets:

| `p` | Depth | Feasibility rate | Optimum hit probability | Hit given feasible |
| --- | --- | --- | --- | --- |
| 1 | 11 | 0.4479 [0.0762, 0.5859] | 0.0881 [0.0117, 0.1309] | 0.1937 [0.1330, 0.2393] |
| 2 | 18 | 0.6027 [0.1875, 0.8828] | 0.1141 [0.0312, 0.1953] | 0.1850 [0.1523, 0.2372] |
| 3 | 25 | 0.5900 [0.0703, 0.9648] | 0.1168 [0.0078, 0.2109] | 0.1795 [0.0676, 0.2365] |

Gate counts scale linearly with depth: `p = 1` is 10 `rzz`, 5 `rz`, 5 `rx`, 5 `h`, 5 `measure`.

### What this shows

**QAOA learns the constraint.** Feasibility rises from the uniform 0.1562 to 0.4479 at `p = 1` and
0.6027 at `p = 2` — roughly a 3–4× improvement. The one-hot penalty is large and uniform across
candidates, so the cost layer suppresses infeasible states effectively.

**QAOA does not learn the objective.** Conditioned on producing a feasible state, the probability of
selecting the optimal candidate is 0.1937, 0.1850, and 0.1795 at `p = 1, 2, 3` — **at or slightly
below the uniform 0.2000 at every depth**. Increasing depth does not improve it; the point estimate
declines monotonically while its spread widens. Within the feasible subspace this sampler is
indistinguishable from a coin flip among the five candidates.

This is the near-degeneracy predicted in
[QUBO-to-Ising conversion](qubo_formulation.md#qubo-to-ising-conversion). All ten couplings are
identical at `J = 1.0`, and the local fields — the only candidate-specific information in the
Hamiltonian — span `-3.0296` to `-3.0585`, a spread under 1% of their magnitude. The penalty
structure dominates the cost Hamiltonian so completely that the objective is nearly invisible to it.

**The zero optimality gap is an artifact of shot count, not solver quality.** `objective_value` is
the best objective over all samples, so with 512 shots and a ~9% per-shot hit rate the optimum
appears somewhere in essentially every run, and `relative_optimality_gap` is 0.0 in all 30 runs.
Reporting that as success would be misleading. Best-of-512 sampling from a five-element set finds
the minimum whether or not the sampler is informed. `optimum_hit_probability` is the honest metric
here, and the ratio against uniform is the honest comparison.

**Seed variance is large and grows with depth.** At `p = 3` feasibility ranges from 0.0703 to 0.9648
across ten seeds — a 14× spread. COBYLA is a local optimizer started from a random point, and deeper
circuits give it a rougher landscape. No single run from this configuration is meaningful, and a
favourable seed at `p = 3` could be presented as a strong result if the other nine were discarded.
Do not do this.

### Single reproducible run

Seed 42, `p = 1`, 512 shots, local Aer:

```
feasibility_rate         0.5625
objective_value          0.059218477950031836
relative_optimality_gap  0.0
optimum_hit_probability  0.087890625
stability                0.2569444444444444
runs                     512
circuit_depth            11
gate_counts              {'rzz': 10, 'h': 5, 'rz': 5, 'rx': 5, 'measure': 5}
parameters               [2.861222, 1.162882]
```

### Conclusion

On this instance QAOA reproduces the constraint and not the objective, and it is beaten on every
metric by an O(n) exact enumeration that runs in microseconds. **This is not evidence of quantum
advantage, and it is not evidence against it.** It is a five-variable instance with a near-degenerate
feasible spectrum, which is the wrong regime in which to look for either. The value of the result is
that the implementation is verified end to end and that the degeneracy problem is now measured rather
than assumed. A meaningful comparison requires the harder formulations in
[`development_plan.md`](development_plan.md#stage-7--stronger-research-extension).

## Reproducing a run

```bash
pip install -r requirements.txt
python - <<'PY'
from qkash.scoring import build_selection_qubo, verify_qubo_equivalence
from qkash.quantum import run_qaoa
from qkash.benchmark import summarize_samples
from qkash.scoring import solve_original_exact

scores = [
    0.059218477950031836,
    0.06225822738678684,
    0.07129763664836268,
    0.09071402607098497,
    0.11691012797116852,
]
qubo = build_selection_qubo(scores, penalty=2.0)
assert verify_qubo_equivalence(qubo)["equivalent"]

result = run_qaoa(qubo, reps=1, shots=512, max_qubits=5, seed=42, backend_name="local_aer")
exact = solve_original_exact(scores)
print(result["status"], result["execution_backend"])
print("depth", result["circuit_depth"], "gates", result["circuit_gate_counts"])
print(summarize_samples("QAOA", result["samples"], float(exact["objective"]),
                        exact["indices"], float(result["runtime_s"])))
PY
```

Record the seed with every result. Vary it across at least 10 runs and report the distribution, not
a single favourable draw.

To run against qBraid, set `QBRAID_API_KEY` and `QBRAID_DEVICE_ID` in `.env` and pass
`backend_name="qbraid"`. See [`README.md`](../README.md#qbraid) for configuration.

## Limitations

- **Five qubits is a demonstration, not evidence.** The instance is solved exactly in microseconds by
  enumeration. A QAOA result here validates the implementation; it says nothing about advantage.
- **The feasible states are nearly degenerate, and this is measured.** Their scores span 0.059 to
  0.117 while the local fields differ by under 1%. Conditioned on feasibility, QAOA selects the
  optimum at or below the uniform 1-in-5 rate at every depth tested. Depth does not fix it.
- **The local backend is noiseless.** Aer results carry no gate error, decoherence, or readout error,
  so they are an upper bound on what hardware would produce.
- **A single run proves nothing.** COBYLA is a local optimizer from a random start; different seeds
  find different parameters. At `p = 3` feasibility varied 14× across ten seeds. Report
  distributions.
- **Never present QAOA output as a commercial recommendation.** It selects a row from a candidate set
  whose limitations are documented in
  [`classical_model.md`](classical_model.md#limitations), including the mixed-period defect.
