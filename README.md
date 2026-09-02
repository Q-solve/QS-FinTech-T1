# QKash

QKash optimizes East Africa remittance provider selection from the RPW CSV data.

## Flow

1. Load `data/rpw_dataset_2011_2025_q3_EastAfrica.csv`.
2. Apply user filters before Pareto pruning.
3. Convert the priority query into three objective weights: transaction fee,
   time, and FX spread.
4. Calculate normalized weighted scores.
5. Pareto prune dominated candidates.
6. Build the constrained one-hot model and equivalent QUBO.
7. Verify QUBO equivalence before running QAOA.
8. Compare business-as-usual, exact, simulated annealing, and QAOA outcomes.

## Run

```bash
/home/thairu/python4code/bin/pip install -r requirements.txt
/home/thairu/python4code/bin/python -m streamlit run app.py
```

In VS Code, select `/home/thairu/python4code/bin/python` as the Python
interpreter. This repository intentionally does not keep a project-local
virtual environment.

## Metrics

QKash reports feasibility rate, objective value, relative optimality gap,
optimum-hit probability, end-to-end runtime, stability, and time to solution.

## qBraid

Copy `.env.example` to `.env` or export the variables in your shell.
`.env.example` is a template only and is not read for secrets.

QAOA uses this execution order:

1. Optimize gamma and beta locally with Qiskit Aer.
2. Construct the optimized measured circuit.
3. Execute the final circuit with either `LocalAerBackend` or `QBraidBackend`.

The app never submits qBraid jobs during parameter optimization. qBraid is only
used for the final optimized circuit when the qBraid backend is selected.
