# Remit-Q project instructions

These instructions apply to the entire repository.

## Project purpose

Project title: **Quantum-Assisted Multi-Objective Optimization of Cross-Border Remittance Services**.

Build an evidence-backed comparison of quantum and classical approaches for recommending cross-border remittance provider/service alternatives according to user-defined preferences for cost and speed. Never assume or claim quantum advantage; experiments must determine the outcome.

## Verified repository and environment

- Repository root: `/Users/iannganga/Documents/remit-q`.
- Git branch: `main`. Preserve unrelated and uncommitted user files.
- Use the existing `.venv`; it was verified as Python 3.11.16.
- The existing frontend is a React/TypeScript/Vite starter under `frontend/`. Do not assume other components have been scaffolded.
- The raw workbook exists at `data/raw/rpw_dataset_2011_2025_q3_EastAfrica.xlsx`. Its verified SHA-256 is `7d3b394c0db9a8c4227cccd52bb73e09c026f3f1a74de6befdee8b6477ab1ce4`; never modify or commit it.
- The audited derived file is `data/processed/remittance_east_africa_clean.csv`.

## Data safety and provenance

- Never modify the raw workbook.
- Before data work, read `docs/data_audit.md`.
- Read the workbook with `openpyxl` in read-only/data-only mode and explicitly restrict the logical range to A:AD (`min_col=1`, `max_col=30`). Do not materialize all 16,384 worksheet columns.
- Treat row 1 as the header, row 2 as a blank spacer, and rows 3–4506 as 4,504 logical records.
- The worksheet is physically malformed from rows 972–4506: each logical 30-field record is repeated horizontally toward XFD. Only A:AD is the authoritative logical record.
- Preserve all 30 source fields in derived datasets and document every cleaning operation, row-count change, schema change, imputation, and standardization.
- Do not invent columns, providers, corridors, measurements, or results.
- Keep raw provider names unless an explicit, documented mapping is approved. Case/whitespace normalization is safe for comparison, but do not merge names merely because they appear related. In particular, do not automatically merge `Ecobank` with `EcoBank Rapid Transfer`, or `United Bank for Africa (UBA)` with `UBA Africash`.
- Distinguish true nulls from source category strings such as `N/A`.
- Do not silently discard negative FX margins or costs, the 19 known CC2 total-cost inconsistencies, or the four known missing CC2 metric sets. Define eligibility rules and report exclusions.

## Modeling rules

- Filter by period, source, destination, amount/benchmark, and service eligibility before creating decision variables. Dataset rows are observations, not qubits or decision variables.
- For the first Kenya→Tanzania MVP, start from the 14 exact-unique 2025_3Q provider/service alternatives documented in `docs/data_audit.md`, subject to explicit eligibility rules.
- Zanzibar is treated as Tanzania for the demonstration because the workbook contains no Zanzibar value.
- The demonstration amount KES 100,000 is not directly quoted by the 2025_3Q data, which contains KES 18,000 and KES 45,000 benchmark amounts. Do not present an exact KES 100,000 quote without a documented interpolation, extrapolation, or external fee schedule.
- `CC1 TOTAL COST %` and `CC2 TOTAL COST %` generally equal fee/amount × 100 plus the corresponding FX margin. Do not add FX margin to total cost again.
- If fee and FX margin are separate objectives, do not simultaneously include total cost as an independent cost objective without explicitly controlling double counting.
- Normalize objectives consistently within the same eligible comparison set. Record the normalization method, bounds, direction, treatment of constants/outliers, and weight semantics.
- Provider selection and network routing are separate formulations. The workbook supports direct-corridor provider/service selection. It does not contain route sequences, intermediate transfers, capacities, edge compatibility, or routing feasibility, so it does not by itself support true network routing.
- Dijkstra is relevant only for a genuine graph-routing extension with defensible edge data.
- A one-of-N provider selection is classically trivial. State this plainly. A stronger research extension should add multiple transactions, corridor or provider capacities, budgets, service constraints, diversification/fairness, or a validated multi-leg network.

## Quantum/classical comparison

- Implement exact/exhaustive and mathematical-optimization baselines before interpreting QAOA results.
- Use the same objective, constraints, eligible instances, normalization, and hardware/runtime accounting across methods.
- Measure solution quality, feasibility rate, runtime, optimality gap, success probability, and scaling behavior.
- Separate circuit construction, transpilation, optimization-loop, sampling, queue, and end-to-end time where applicable.
- Use repeated seeds/runs and report distributions or confidence intervals, not a single favorable run.
- Never display fabricated quantum savings, advantage, or hardware results in code, docs, tests, or the UI.

## Planned implementation stack

- Data/backend: Python 3.11, pandas, openpyxl, FastAPI, pytest.
- Optimization: Qiskit, Qiskit Aer, Qiskit Optimization, QAOA, exact enumeration, and a mathematical-optimization baseline.
- Frontend: React, TypeScript, Vite, Tailwind CSS, Lucide icons, and Recharts.
- Later execution: qBraid simulator/hardware after local correctness and reproducibility are established.

Use the existing `.venv`. Inspect installed packages before adding dependencies, and ask before installing when the required package set or version compatibility is uncertain.

## Engineering workflow

- Inspect before editing; preserve user work and avoid destructive Git operations.
- Keep data preparation, domain logic, solvers, experiment harnesses, API schemas, and UI concerns separated.
- Prefer deterministic, testable functions and versioned experiment configurations.
- Add tests for filtering, deduplication, cost identities, normalization, one-hot feasibility, objective equivalence, and API validation.
- Cite dataset-derived UI values to a period/corridor/benchmark and expose limitations rather than hiding them.
- Do not implement the solver, API, or frontend until the relevant stage in `docs/development_plan.md` is approved.
