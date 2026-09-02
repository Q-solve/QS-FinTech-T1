# Constraint-preserving XY-QAOA experiment

## Motivation and unchanged optimization target

The standard penalty-based QAOA experiment starts in the uniform superposition
over all `2^14` binary states and uses the standard X mixer. A single X action
changes a bit and therefore changes Hamming weight. Although the validated QUBO
penalizes states that violate `sum_i x_i = 1`, neither the initial state nor the
standard mixer confines evolution to the feasible subspace. In the recorded
standard p=1 experiment, every raw modal state was infeasible and mean feasible
probability was about 0.232324.

The constraint-preserving variant changes only the initial state and mixer. It
continues to consume the existing `QuboModel.qubo_problem.to_ising()` cost
Hamiltonian with the same alternatives, variable order, normalized scores,
weights, and penalty `P = 2.0`. For every feasible one-hot state,
`P(sum_i x_i - 1)^2 = 0`. In ideal evolution, the XY formulation cannot reach
the penalized infeasible subspace, but retaining the penalty keeps the cost
Hamiltonian identical for direct comparison.

## Ring XY mixer

For periodic edges

```text
(0,1), (1,2), ..., (12,13), (13,0)
```

the mixer Hamiltonian is

\[
H_M=\sum_{(i,j)\in E}(X_iX_j+Y_iY_j).
\]

Let the excitation-number operator be

\[
N=\sum_i\frac{I-Z_i}{2}.
\]

Each `XX + YY` term coherently exchanges `|10>` and `|01>` while leaving
`|00>` and `|11>` unchanged. It moves an excitation rather than creating or
destroying one, so `[X_iX_j+Y_iY_j, N]=0` and therefore `[H_M,N]=0`. The code
constructs both operators independently and verifies their commutator.

The QAOA circuit implements a deterministic first-order product formula around
the ring using Qiskit's `XXPlusYYGate`. Each edge evolution individually
preserves excitation number; their ordered product therefore does too even
where adjacent edge terms do not commute. Connectivity is stored in each run.

## Uniform one-hot W initialization

The initial state is

\[
|W_{14}\rangle=\frac{1}{\sqrt{14}}\sum_{i=0}^{13}|x_i=1, x_{j\ne i}=0\rangle.
\]

The displayed ket above follows documented `x_0 ... x_13` variable order.
Qiskit state labels display the reverse order, `x_13 ... x_0`; an excitation at
`x_0` is therefore label `00000000000001`, while an excitation at `x_13` is
`10000000000000`.

The implementation does not use a dense 16,384-amplitude initializer. It starts
with an X gate on qubit 0 and applies `n-1` nearest-neighbor XX+YY rotations.
At each step the circuit leaves amplitude `1/sqrt(n)` behind and passes the
remaining excitation amplitude to the next qubit. A fixed phase parameter
produces equal positive amplitudes. This O(n)-gate construction is suitable for
later decomposition and transpilation, although nonlocal hardware connectivity
may still add routing overhead.

Before QAOA execution, independent statevector checks verify normalization,
equal probability `1/14` on all feasible states, zero initial infeasible
probability, variable/label mapping, mixer Hermiticity, zero number-operator
commutator, and zero leakage after several mixer angles on 14 qubits.

## Reference experiments

Both XY experiments use the same reference instance and fixed protocol as the
standard results: `2025_3Q`, `KENTZA`, CC2/KES 45,000, weights
`(0.4, 0.3, 0.3)`, penalty 2.0, 2,048 shots, COBYLA with at most 60 evaluations,
and seeds 11, 29, 47, 71, and 97. No seed was tuned separately.

| Experiment | Seeds | Mean feasible probability | Mean leakage | Optimum recovery | Mean optimal-state probability | Mean best-feasible gap | Modal feasible | Depth original / decomposed / transpiled | Mean runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard X, p=1 | 5 | 0.232324 | 0.767676 | 3/5 (60%) | 0.023438 | 0.119497 | 0/5 | 28 / 78 / 28 | 0.522 s |
| XY + W, p=1 | 5 | 1.000000 | 0.000000 | 5/5 (100%) | 0.097656 | 0.000000 | 5/5 | 32 / 283 / 72 | 0.447 s |
| Standard X, p=2 | 5 | 0.010547 | 0.989453 | 3/5 (60%) | 0.002344 | 0.000000 among feasible runs | 0/5 | 44 / 122 / 44 | 0.947 s |
| XY + W, p=2 | 5 | 1.000000 | 0.000000 | 5/5 (100%) | 0.059082 | 0.000000 | 5/5 | 61 / 469 / 115 | 0.860 s |

For XY p=1, individual optimal-state probabilities were 0.095215, 0.113281,
0.088867, 0.090332, and 0.100586. For XY p=2 they were 0.117676, 0.047852,
0.006348, 0.053223, and 0.070312. All ten XY modal states were feasible, and
all ten distributions sampled the exact optimum at least once. The exact
optimum was not the modal state in any XY run. Thus the experiment shows a
large improvement in feasibility and finite-shot optimum recovery, but it does
not show that the optimizer concentrated most probability on the optimum.

The p=2 standard experiment was rerun across the same five seeds as XY. Its
exact optimum was sampled in three runs, but no standard-X modal state was
feasible. Increasing p also did not increase the XY mean optimal probability
here: it fell from about 0.097656 at p=1 to 0.059082 at p=2.

## Interpretation and limitations

The XY formulation eliminates ideal constraint leakage by construction. This is
an improved quantum formulation for this exactly-one constraint and an observed
improvement in simulator feasibility. It is not quantum advantage over classical
computing. The underlying one-of-14 selection remains classically trivial after
scores are computed, and direct argmin remains the meaningful baseline.

The XY circuits are materially deeper after decomposition and transpilation.
Recorded local runtimes are experimental classical-simulator timings affected by
optimizer paths, implementation overhead, and host load; the small p=1 timing
difference is not a speedup claim. A real device may introduce gate error,
decoherence, routing overhead, and measured leakage even though the ideal mixer
preserves Hamming weight. Later hardware work must remeasure feasibility and
leakage and record backend topology, transpilation, calibration, queue time, and
noise-mitigation choices.

Run either formulation with the shared CLI:

```bash
.venv/bin/python -m backend.scripts.run_qaoa \
  --mixer constraint_preserving_xy \
  --reps 1 \
  --shots 2048 \
  --max-iterations 60 \
  --seeds 11 29 47 71 97 \
  --output experiments/results/qaoa_xy_p1.json
```

The CLI refuses to overwrite an existing result unless `--overwrite` is
explicitly supplied. New expanded experiment and comparison artifacts use
schema version `2.0`; the comparison reader explicitly supports the preserved
legacy standard p=1 artifact at schema `1.0`. The legacy artifacts remain
unchanged.
