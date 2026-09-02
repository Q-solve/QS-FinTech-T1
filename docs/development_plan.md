# Staged development plan

## Guiding outcome

Build a reproducible research prototype that recommends an eligible direct remittance
provider/service and compares QAOA against appropriate classical baselines without claiming an
advantage in advance. The current five-qubit Kenya → Tanzania instance is a correctness and
demonstration case; scaling experiments and stronger constrained formulations carry the research
value.

## Stage gates and claims policy

A stage is complete only when its exit criteria are met and its document is updated. No stage may
publish a quantum-advantage claim. Any comparative statement must name the instance size, the seed,
the backend, and the number of runs behind it.

## Stage 0 — Data foundation

**Status: audited, pipeline outstanding.**

Delivered:

- Full audit of the 4,504-row, 39-column dataset in [`data_audit.md`](data_audit.md), with SHA-256
  values recorded for both source files.
- Deterministic load and normalization path in `qkash/data.py`.
- Both CSVs excluded from version control.

Outstanding:

- Turn the one-off audit into an automated guardrail: assert row count, column set, corridor/code
  agreement, and the CC1 cost identity on load.
- Emit a machine-readable manifest recording source hash, row counts, and cleaning version.

Exit criteria:

- Loading an unmodified CSV reproduces 4,504 rows and 41 columns.
- Tests confirm zero exact duplicates, corridor codes agreeing with source and destination codes, and
  the CC1 cost identity holding within 0.01 percentage points.

## Stage 1 — Data defect remediation

**Status: not started. Blocking for any recency-dependent result.**

Three open High-severity defects from the audit risk summary:

- **Date parsing.** 246 rows in `2025_1Q` and `2025_3Q` fail the fixed `%d/%b/%Y` format and receive
  `NaT`, corrupting `latest_per_firm` and the business-as-usual baseline. Accept both the
  `24/Jan/2011` and ISO forms.
- **Coverage vocabulary.** `coverage_score` maps only Nationwide / Major cities / Main city, leaving
  3,535 rows (78%) on the 0.35 default. Extend the mapping to High / Medium / Low, or make the
  function raise on an unrecognized value rather than silently defaulting.
- **Dead `pickup method` filter option.** `unique_values` splits values on commas to build the
  dropdown, so `Mobile, Cash` contributes a standalone `Mobile` option, but `_filter_exact` compares
  against the whole raw string and never matches it. Route `pickup method` through `_filter_token`
  as `payment instrument` and `access point` already are.

Also in scope, at Medium severity:

- Normalize `pickup method` casing for comparison while preserving the raw label for display, so
  `Bank account` and `Bank Account` stop fragmenting the filter.

Exit criteria:

- Zero `NaT` values in `date_parsed` for a clean load.
- No candidate carries a defaulted `coverage_score` without an explicit flag.
- Every option `unique_values` offers for a field returns at least one row when selected.
- Tests cover both date formats, every coverage vocabulary value, and dropdown/filter agreement.

## Stage 2 — Objective and normalization specification

**Status: implemented and specified in [`classical_model.md`](classical_model.md).**

Three objectives — transaction fee, transfer time, FX spread — normalized to `[0, 1]` losses over
the filtered pool, combined by weights summing to one. `total_cost_pct` is deliberately excluded to
avoid double-counting fee and FX margin.

Outstanding:

- **Fix the keyword-weighting quirk.** The shipped default query yields 0.2867 / 0.2867 / 0.4266
  rather than balanced weights, because the three keyword lists differ in length. Either equalize the
  lists, normalize each objective's hits by its list length, or make the default query use explicit
  overrides.
- Decide whether normalization should be computed over the filtered pool or the modelled subset, and
  document the choice with its effect on comparability across runs.

## Stage 3 — Classical baselines

**Status: implemented.**

Business as usual, exact enumeration, and simulated annealing are in `qkash/classical.py`, all
returning a common result shape. `solve_original_exact` provides ground truth in O(n) and returns
the full co-optimal set.

Outstanding:

- Add a mathematical-programming baseline (an LP/MIP formulation) so the comparison includes a
  solver class that scales beyond enumeration.

## Stage 4 — QUBO and QAOA implementation

**Status: implemented and verified.**

The QUBO, its penalty condition, the Ising conversion, and the exhaustive equivalence check are
specified in [`qubo_formulation.md`](qubo_formulation.md) and verified on the reference instance.
QAOA is specified in [`qaoa_experiment.md`](qaoa_experiment.md).

Exit criteria met:

- `verify_qubo_equivalence` gates every run and `app.py` aborts on failure.
- `validate_solver_outputs` gates every metric.
- The five-qubit reference instance shows all feasible states below 0.117 and all infeasible states
  at or above 2.0.
- QAOA reference results across 10 seeds at `p = 1, 2, 3` are recorded in
  [`qaoa_experiment.md`](qaoa_experiment.md#reference-results). They show QAOA reproducing the
  one-hot constraint but not discriminating among feasible candidates, which is the expected
  consequence of the near-degenerate spectrum.

## Stage 5 — Experiment and scaling framework

**Status: not started. This is where the research value begins.**

Currently every run is a one-off through the UI. Required:

- A headless experiment runner taking a versioned configuration — filter spec, weights, tier,
  candidate count, penalty multiplier, `p`, shots, seed list — and writing results to
  `experiments/results/`.
- Repeated seeds with reported distributions or confidence intervals, never a single run.
- A scaling sweep over candidate count and `p`, recording where QAOA quality degrades.
- Separate accounting for circuit construction, transpilation, optimization loop, sampling, and
  queue time. `run_qaoa` already separates `runtime_s` from `backend_runtime_s`; extend this.

Exit criteria:

- Any published figure is reproducible from a stored configuration and seed list.
- No result in the repository comes from a single run.

## Stage 6 — qBraid and hardware execution

**Status: implemented for simulators, unexercised on hardware.**

`QBraidBackend` submits only the final optimized circuit, per the policy in `AGENTS.md`. The default
device is the `qbraid:qbraid:sim:qir-sv` simulator.

Outstanding:

- A recorded end-to-end qBraid simulator run with its job id.
- Hardware execution, only after Stage 5 establishes local reproducibility.
- Explicit accounting for queue time, which currently falls inside `end_to_end_runtime_s`.

## Stage 7 — Stronger research extension

**Status: not started.**

One-of-N provider selection is classically trivial and cannot support a research claim. A defensible
extension must add genuine combinatorial structure. Candidates, in rough order of data feasibility:

- **Multiple simultaneous transactions** across corridors sharing a budget.
- **Provider or corridor capacity constraints**, making selections interact.
- **Diversification or fairness constraints** — no more than `k` transfers through one provider.
- **Multi-period planning** over the 54 available quarters.

A true network-routing formulation is **not** available from this dataset. As recorded in
[`data_audit.md`](data_audit.md#not-supported-true-network-routing), it contains no route sequences,
intermediate legs, capacities, or edge compatibility. Do not build a routing narrative on it.

## Stage 8 — Presentation and reporting

**Status: partially implemented in `app.py`.**

The UI surfaces filtered/scored/Pareto/model counts, the fallback-candidate notice, verification
status, the metric table, selected solutions, the circuit diagram, and the candidate set.

Outstanding:

- Surface the data-quality flags from Stage 1 next to any affected result.
- Show the period span of the modelled candidate set, so a user sees when a comparison mixes
  quarters.
- Cite every displayed value to a period, corridor, and benchmark tier.
