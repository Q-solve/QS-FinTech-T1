# Local QAOA experiment

## Scope and source model

This stage evaluates the Quantum Approximate Optimization Algorithm (QAOA) on
the already validated Remit-Q QUBO. It does not rebuild or alter filtering,
alternatives, normalization, weights, speed encoding, variable order, QUBO
coefficients, or the exactly-one penalty. The implementation calls
`QuboModel.qubo_problem.to_ising()` and therefore consumes the same model used
by the exact QUBO tests and solvers.

The reference experiment uses `PERIOD = 2025_3Q`, corridor `KENTZA`, benchmark
`CC2` (KES 45,000), fee/FX/speed weights `(0.4, 0.3, 0.3)`, penalty `P = 2.0`,
and 14 binary variables/qubits. The dataset does not quote KES 100,000, and
this experiment does not interpolate or extrapolate to that amount.

Before QAOA runs, three exact baselines are checked on the same instance:
direct weighted argmin, independent QUBO enumeration, and the constrained
binary program solved by SciPy/HiGHS through Qiskit Optimization. The artifact
records the mathematical solver's status, feasibility, dual bound, MIP gap,
node count, and runtime. Execution stops if this baseline disagrees with the
direct classical optimum.

## Algorithm

The cost Hamiltonian is the Ising operator returned by Qiskit from the
validated unconstrained QUBO

\[
E(x)=\sum_i c_i x_i + P\left(\sum_i x_i-1\right)^2.
\]

Qiskit returns the Ising operator and a scalar offset. The optimizer callback
records expectations of the operator without that offset; decoded sample
energies use the full QUBO convention and include its constant term. The
offset changes reported energy values but not the minimizing state.

At depth `p` (called `reps` by Qiskit), standard QAOA alternates `p` cost
evolutions with `p` evolutions under the standard sum-of-X mixer. The mixer is
left at Qiskit's standard default; no feasibility-preserving custom mixer or
warm start is used. The initial implementation uses `p = 1`, producing two
variational circuit parameters. COBYLA adjusts those parameters to minimize a
finite-shot estimate of the Hamiltonian expectation.

After optimization, the circuit is sampled again at the optimized parameters.
QAOA is approximate: finite depth restricts the reachable state family,
finite shots introduce sampling variation, and a local classical optimizer can
finish at a suboptimal parameter point. A sampled exact optimum is therefore
an experimental outcome, never a test precondition.

## Bit order and sample interpretation

The validated variable vector is written `x_0 ... x_(n-1)`. Qiskit displays a
computational-basis label in the reverse order, `x_(n-1) ... x_0`. Every
reported state retains both strings. For the exact reference optimum these are
`00000000000010` in variable order and `01000000000000` as the Qiskit label.

Each run reports these quantities separately:

- the raw most-probable measured state and its feasibility;
- the lowest-score feasible state that actually received nonzero samples;
- the empirical probability of every exactly-one state and their total
  feasible probability;
- the combined probability of all exact optimal exactly-one states;
- whether an exact optimum appeared at least once and whether one was the raw
  most-probable state;
- the best feasible weighted score, full QUBO energy, and absolute gap from
  the exact feasible optimum;
- the provider, payment instrument, and receiving method only when the decoded
  state is feasible.

Finding a best feasible sampled state is analysis of the observed distribution,
not a replacement for the raw mode. If no exactly-one state is sampled, the run
is explicitly unsuccessful and returns no recommendation. No infeasible sample
is silently repaired. An approximation ratio is intentionally omitted because
the reference optimum is close to zero, making that ratio unstable and
misleading.

## Reference configuration and reproducibility

The initial full experiment uses:

| Setting | Value |
|---|---:|
| QAOA repetitions | 1 |
| Shots per circuit evaluation | 2,048 |
| Optimizer | COBYLA |
| Maximum optimizer evaluations | 60 |
| Algorithm, simulator, and transpiler seeds | 11, 29, 47, 71, 97 |
| Transpiler optimization level | 1 |
| Simulator | local, noiseless Qiskit Aer `SamplerV2` |

For each run, the same documented seed initializes QAOA parameters, Aer's
finite-shot sampling, and transpilation. The initial point is generated
explicitly from NumPy's seeded generator within the ansatz parameter bounds.
The JSON artifact records Python and package versions, all experiment settings,
each run, every optimization callback evaluation, and aggregate statistics.
Standard deviations are population standard deviations over the configured
set of seeds. Gap statistics include only runs in which a feasible state was
sampled; the aggregate separately reports the number of such successful runs.
The artifact also records the audited raw-data hash, a freshly computed
processed-CSV hash, normalization rules and bounds, and every alternative's
raw objectives, normalized components, and weighted score in variable order.

The command rejects a penalty at or below the instance's documented safe
threshold. Such a penalty would make an infeasible state an unconstrained QUBO
optimum while the experiment's recovery metrics refer to the exactly-one
classical optimum. Rejecting it prevents those two targets from being mixed.

Run the reference experiment from the repository root:

```bash
.venv/bin/python -m backend.scripts.run_qaoa \
  --benchmark CC2 \
  --fee-weight 0.4 \
  --fx-weight 0.3 \
  --speed-weight 0.3 \
  --penalty 2.0 \
  --reps 1 \
  --shots 2048 \
  --max-iterations 60 \
  --seeds 11 29 47 71 97 \
  --output experiments/results/qaoa_reference_p1.json
```

The command-line options also permit a documented exploratory `p = 2` run.
That run is optional because its additional circuit depth and optimization
cost can be prohibitive on the local simulator.

## Metrics and limitations

Every run records qubit count, repetitions, shots, all seeds, optimizer and
limit, actual evaluations, original/decomposed/transpiled circuit depth, and
parameter count. Timings separately cover Ising conversion, circuit
construction/inspection, transpilation, sampler jobs during parameter
optimization, and final sampling. Qiskit's optimizer time and end-to-end local
execution time are also retained. Optimizer time overlaps its sampler-job time
and therefore the components must not be added as if they were disjoint. Local
queue time is recorded as zero because no remote backend is involved. The
end-to-end timer excludes dataset loading, QUBO construction, exact baselines,
process startup, and JSON writing.

These values are experimental implementation timings on a classical computer.
They do not measure quantum hardware performance and are not evidence of
quantum speedup or quantum advantage. The one-of-14 problem is classically
trivial after scoring: direct argmin is linear in 14 alternatives. Aer is
noise-free apart from finite-shot sampling; no hardware queueing, device noise,
error mitigation, or qBraid execution is included. Conclusions from this
small instance are limited to implementation correctness and observed QAOA
sampling behavior under the recorded configuration.
