# QKash project instructions

These instructions apply to the entire repository.

## Project purpose

Project title: **Quantum-Assisted Multi-Objective Optimization of Cross-Border Remittance Services**.

QKash builds an evidence-backed comparison of quantum and classical approaches for recommending a
cross-border remittance provider/service according to user-defined preferences for transaction fee,
transfer time, and FX spread. Never assume or claim quantum advantage; experiments must determine the
outcome.

## Verified repository and environment

- Application entry point: `app.py`, a Streamlit UI.
- Library package: `qkash/` — `data`, `scoring`, `classical`, `quantum`, `benchmark`, `qbraid_bridge`.
- Tests: `tests/test_core.py`, run with `pytest`. The tests build their own synthetic DataFrame and
  never read the CSV, so they pass without the dataset present.
- This repository intentionally keeps no project-local virtual environment under version control.
  `.vscode/settings.json` pins `python.defaultInterpreterPath` to an interpreter outside the tree.
- The dataset is `data/processed/remittance_east_africa_clean.csv` — 4,026 rows, 30 columns, an
  audited and deduplicated RPW East Africa export. It is **not tracked by Git**. Obtain it as
  described in `docs/data_audit.md` before running `app.py`.
- `data/QKash.png` is tracked and is the only committed binary asset.

## Data safety and provenance

- Never modify the source CSVs in place. Treat them as read-only inputs.
- Before data work, read `docs/data_audit.md`.
- `qkash/data.py` lowercases and strips every column label on load, and maps a small alias set
  (`access_point` → `access point`, `pick-up method` → `pickup method`, and similar). Do not add new
  column spellings without updating `COLUMN_ALIASES` and the audit document together.
- Dates are parsed by `parse_dates`, which tries each format in `DATE_FORMATS` against the values
  still unparsed. Never narrow this back to a single fixed format: the audited export is ISO and the
  older extracts are `24/Jan/2011`, and a single format silently produces `NaT` rather than failing.
- Do not invent columns, providers, corridors, measurements, or results.
- Keep raw provider names unless an explicit, documented mapping is approved. Case and whitespace
  normalization is safe for comparison, but do not merge names merely because they appear related. In
  particular, do not automatically merge `Ecobank` with `EcoBank Rapid Transfer`, or
  `United Bank for Africa (UBA)` with `UBA Africash`.
- Do not silently discard negative FX margins or costs, the 19 known CC2 total-cost inconsistencies,
  or the four known missing CC2 metric sets. Define eligibility rules and report exclusions.
- `prepare_candidates` drops rows with a null `firm`, `amount_lcu`, `fee_lcu`, `fx_margin`, or
  `total_cost_pct`. Any change to that rule changes the eligible comparison set and must be recorded
  in `docs/data_audit.md`.

## Modeling rules

- Filter corridor, payment instrument, pickup method, and amount tier **before** scoring and Pareto
  pruning. `tests/test_core.py::test_filtering_happens_before_pareto_pruning` enforces this ordering.
  Dataset rows are observations, not qubits or decision variables.
- The three supported objectives are transaction fee, transfer time, and FX spread. All three are
  encoded as lower-is-better losses in `qkash/scoring.py`.
- `receiving network coverage` and `access point` do not exist in the current export and are created
  empty by the loader. Do not build an objective or a filter on either without sourcing the field
  first; `coverage_score` is constant at 0.35 across every row.
- Min-max normalization is computed over the filtered candidate pool, not over the final modelled
  subset. Record the normalization method, bounds, direction, treatment of constants and outliers,
  and weight semantics whenever they change.
- `TARGET_QUBITS` in `app.py` fixes both the candidate count and the QAOA qubit count at 5. A run
  that cannot supply 5 real candidates must fail loudly rather than pad the model with synthetic rows.
- Every qubit must map to a real candidate row. When strict Pareto pruning yields fewer than
  `TARGET_QUBITS` rows, `select_qubo_candidates` fills the remaining slots with next-best scored rows
  and labels them `Best scored fallback` in the `model_source` column. That label must stay visible
  in the UI.
- Provider selection and network routing are separate formulations. This dataset supports
  direct-corridor provider/service selection. It contains no route sequences, intermediate transfers,
  capacities, or edge compatibility, so it does not by itself support true network routing.
- A one-of-N provider selection is classically trivial. State this plainly. A stronger research
  extension should add multiple transactions, corridor or provider capacities, budgets, service
  constraints, diversification/fairness, or a validated multi-leg network.

## Quantum/classical comparison

- Never report QAOA output before `verify_qubo_equivalence` returns `equivalent = True`. `app.py`
  aborts the run when it does not.
- Never report metrics before `validate_solver_outputs` passes. Every sample must have a bit vector of
  the right length, a feasibility flag consistent with its Hamming weight, and an objective that
  matches the candidate score it claims.
- Use the same objective, constraints, eligible instances, normalization, and runtime accounting
  across all four solvers: business as usual, exact mathematical baseline, simulated annealing, QAOA.
- Measure feasibility rate, objective value, relative optimality gap, optimum-hit probability,
  end-to-end runtime, stability, and time to solution. `qkash/benchmark.py` is the single source of
  truth for these definitions.
- Report distributions across repeated seeds, not a single favorable run.
- Never display fabricated quantum savings, advantage, or hardware results in code, docs, tests, or
  the UI. A skipped or failed QAOA run must surface as `status = "skipped"` or `"failed"` with its
  note, not as a silent omission.

## Implementation stack

- Data and UI: Python, pandas, numpy, Streamlit.
- Optimization: Qiskit, Qiskit Aer, SciPy COBYLA, exhaustive enumeration, Metropolis simulated
  annealing.
- Remote execution: qBraid, used only for the final optimized circuit.

Inspect installed packages before adding dependencies, and ask before installing when the required
package set or version compatibility is uncertain.

## qBraid execution policy

QAOA uses a fixed three-step execution order, and it is not negotiable:

1. Optimize gamma and beta locally with Qiskit Aer.
2. Construct the optimized measured circuit.
3. Execute that final circuit once, on either `LocalAerBackend` or `QBraidBackend`.

The app never submits qBraid jobs during parameter optimization. Secrets are read from `.env` only;
`.env.example` is a tracked template and is never read for credentials. `qbraid_status()` must stay
secret-free — it reports whether a key is present, never the key.

## Engineering workflow

- Inspect before editing; preserve user work and avoid destructive Git operations.
- Keep data preparation, scoring, solvers, benchmarking, and UI concerns in separate modules.
- Prefer deterministic, testable functions. Every solver accepts a seed and must honour it.
- Add tests for filtering order, Pareto pruning, QUBO equivalence, candidate backfill, solver-output
  validation, and metric ranking.
- Cite dataset-derived UI values to a period, corridor, and benchmark tier, and expose limitations
  rather than hiding them.
