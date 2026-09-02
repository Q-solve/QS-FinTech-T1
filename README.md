# QKash

QKash recommends a cross-border remittance provider/service for East African corridors, and compares
QAOA against classical baselines on the same model. It is a research prototype: it does not claim
quantum advantage, and its experiments are designed to test for one rather than assume it.

## Documentation

| Document | Contents |
| --- | --- |
| [`docs/data_audit.md`](docs/data_audit.md) | What the dataset contains, which fields are read, and every known defect |
| [`docs/classical_model.md`](docs/classical_model.md) | Legacy detailed model notes; current flow is summarized below |
| [`docs/algorithm_flow.md`](docs/algorithm_flow.md) | Current source/destination/amount, ML policy, QUBO, solver flow |
| [`docs/qubo_formulation.md`](docs/qubo_formulation.md) | QUBO construction, penalty condition, Ising conversion, verification |
| [`docs/qaoa_experiment.md`](docs/qaoa_experiment.md) | QAOA algorithm, backends, bit order, metrics, caveats |
| [`docs/development_plan.md`](docs/development_plan.md) | Staged plan, current status, what is outstanding |
| [`AGENTS.md`](AGENTS.md) | Repository-wide engineering and modeling rules |

## Getting the data

**The dataset is not tracked by Git.** `data/**/*.csv` is excluded by `.gitignore`, so a fresh clone
has no data and `app.py` will raise `FileNotFoundError` on startup.

QKash reads one audited file — 4,026 rows, 30 columns, deduplicated from the World Bank Remittance
Prices Worldwide East Africa extract. In this workspace the file is at
`data/remittance_east_africa_clean.csv`. If a processed export is later placed at
`data/processed/remittance_east_africa_clean.csv`, the loader will prefer that path automatically.

```bash
sha256sum data/remittance_east_africa_clean.csv
# 0b8f69c8c516fa9ea38951c80724f701ff2b16f2064e37ec3cc66c3b243fac90
```

See [Source, method, and integrity](docs/data_audit.md#source-method-and-integrity) for provenance
and [Schema and loader behaviour](docs/data_audit.md#schema-and-loader-behaviour) for the two columns
this export does not carry.

The test suite builds its own synthetic DataFrame and does **not** need the CSV. `pytest` passes on
a clean clone.

## Running

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The project Streamlit config advertises the app at `http://qkash:8503`. Add this local host
alias once if the name does not resolve on your machine:

```bash
127.0.0.1 qkash
```

This repository intentionally keeps no project-local virtual environment under version control.
`.vscode/settings.json` pins the interpreter path; adjust it to your own environment.

```bash
pytest
```

## Pipeline

The order is fixed, and filtering before pruning is enforced by test.

1. Load the audited remittance CSV once through Streamlit's data cache.
2. Ask the user only for source country, destination country, and transfer amount.
3. Match the transfer amount to the closest historical RPW amount tier (`cc1` or `cc2`).
4. Filter by source and destination, then build amount-specific fee, FX, and time features.
5. Keep the latest observation for each provider/payment/receiving-method service option.
6. Compute fee, FX-spread, and settlement-time losses, then Pareto prune before any weighting.
7. Use K-Means to identify natural service profiles in the Pareto set.
8. Use a Random Forest classifier trained on those cluster-derived labels to infer the transaction
   use-case profile.
9. Convert the inferred profile into internal objective weights; the user is never asked for weights.
10. Score the remaining candidates, cap the QUBO candidate set for the configured qubit budget, and
    build the constrained one-service model and equivalent QUBO.
11. Verify QUBO equivalence. **A failed check aborts the run before QAOA.**
12. Compare business as usual, exact optimization, lowest-fee/fastest/lowest-FX heuristics,
    simulated annealing, and QAOA.
13. Validate every solver's samples before reporting any metric.

Steps 8 and 10 are gates, not diagnostics. Neither can be bypassed from the UI.

## Metrics

Feasibility rate, objective value, relative optimality gap, optimum-hit probability, end-to-end
runtime, stability, and time to solution. All seven are defined in `qkash/benchmark.py` and computed
identically for all solvers.

The solvers are not measured on equal footing: deterministic baselines produce one sample, while
annealing and QAOA produce distributions. Read
[Comparison caveats](docs/qaoa_experiment.md#comparison-caveats) before ranking them.

## qBraid

Copy `.env.example` to `.env` or export the variables in your shell. `.env.example` is a tracked
template and is never read for secrets.

```
QBRAID_API_KEY=
QBRAID_PROVIDER=qbraid
QBRAID_DEVICE_ID=qbraid:qbraid:sim:qir-sv
QKASH_USE_QBRAID=false
```

QAOA uses a fixed execution order:

1. Optimize gamma and beta locally with Qiskit Aer.
2. Construct the optimized measured circuit.
3. Execute that final circuit once, with either `LocalAerBackend` or `QBraidBackend`.

**The app never submits qBraid jobs during parameter optimization.** qBraid is used only for the
final optimized circuit, and only when the qBraid backend is selected.

## Known limitations

These are documented in full in the linked pages, and are summarized here so they are not
discovered late:

- **The consumer model recommends one service option.** Exact classical optimization remains a very
  strong baseline for one-of-N selection, so the research dashboard tests quantum performance rather
  than assuming quantum advantage.
- **K-Means and Random Forest labels are pseudo-supervised.** The dataset has service attributes but
  not explicit consumer use-case labels, so profiles are inferred from cost, FX, time, and channel
  features.
- **`receiving network coverage` and `access point` are absent from this export.** The loader creates
  them empty, so `coverage_score` is a constant 0.35 and the coverage column in the candidate table
  renders blank. Neither currently affects a score.

## Layout

```
app.py                   Streamlit consumer UI, research dashboard, and run orchestration
qkash/data.py            Loading, normalization, corridor filtering, amount matching
qkash/profiling.py       K-Means service profiles and Random Forest policy inference
qkash/scoring.py         Objective losses, internal scoring, Pareto pruning, QUBO verification
qkash/classical.py       BAU, exact, heuristic, and simulated-annealing solvers
qkash/quantum.py         QAOA, Ising conversion, Aer and qBraid backends
qkash/benchmark.py       Metric definitions and solver-output validation
qkash/qbraid_bridge.py   qBraid configuration and secret handling
tests/test_core.py       Pipeline-order, equivalence, ML policy, validation, and ranking tests
```
