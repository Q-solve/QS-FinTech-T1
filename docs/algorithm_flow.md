# Current QKash Algorithm Flow

This document describes the architecture currently implemented in `app.py` and
the `qkash/` modules.

## Consumer Inputs

The consumer path has three mandatory inputs only:

1. Source country.
2. Destination country.
3. Transfer amount.

The user is not asked for payment instrument, pickup method, firm type, access
point, objective weights, or a priority query.

## Data And Candidate Flow

1. `load_dataset` reads the audited CSV through Streamlit's cached loader.
2. `match_transfer_amount` maps the requested amount to the closest historical
   RPW tier, currently `cc1` or `cc2`.
3. `filter_dataset` filters the dataset by source and destination.
4. `prepare_candidates` builds amount-specific numeric fields:
   `fee_lcu`, `fx_margin`, `total_cost_pct`, and `speed_score`.
5. `latest_service_options` keeps the latest observation for each provider,
   payment instrument, and receiving method.
6. `add_objective_losses` computes lower-is-better fee, FX-spread, and
   settlement-time losses.
7. `pareto_prune` removes services dominated on all three losses before any
   weighting is applied.

## ML Policy Inference

`qkash/profiling.py` runs the internal policy inference:

1. `infer_use_case_policy` adds cash, bank, and mobile-digital indicators.
2. K-Means clusters the Pareto-surviving services into natural service profiles.
3. Cluster centroids are mapped to semantic profiles such as low-cost,
   fast-transfer, balanced, cash-oriented, bank-based, and mobile-digital.
4. A Random Forest classifier is trained on those cluster-derived labels.
5. The classifier's average probability across the current candidate set selects
   the transaction use-case profile.
6. The selected profile determines internal weights for transaction fee,
   FX spread, and settlement time.

Because the dataset has no explicit consumer-use-case labels, the Random Forest
is pseudo-supervised by K-Means output. This is suitable for exploratory
research, not a production credit or eligibility decision.

## Optimization Model

The current constrained model recommends one service option:

```text
minimize    sum_i weighted_score_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

Each binary variable maps to one candidate service and therefore to one QAOA
qubit. `MODEL_CANDIDATE_CAP` limits the maximum number of candidate services
sent to the QUBO.

Before QAOA runs, `verify_qubo_equivalence` checks that the QUBO optimum matches
the exact optimum of the original constrained model. Failure aborts quantum
execution with `QUBO Validation failed`.

## Solver Set

The same candidate model is evaluated by:

- Business-as-usual baseline.
- Exact mathematical baseline.
- Lowest-fee heuristic.
- Fastest-transfer heuristic.
- Lowest-FX heuristic.
- Simulated annealing.
- QAOA with local Qiskit Aer parameter optimization and local/qBraid final
  circuit execution.

Every solver result is validated before metrics are reported.

## Outputs

The consumer tab reports:

- Recommended service provider.
- Payment or transfer method.
- Receiving method.
- Transaction fee.
- FX spread.
- Settlement time.
- Expected overall transfer performance.

The research dashboard reports charts and tables for:

- Feasibility rate.
- Objective value.
- Relative optimality gap.
- Optimum-hit probability.
- End-to-end runtime.
- Stability.
- Time to solution.
- Inferred ML profile and internal policy weights.
- QUBO validation details.
- QAOA circuit details.
