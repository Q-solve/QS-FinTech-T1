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
   `fee_lcu`, `fx_margin`, `total_cost_pct`, `speed_score`, and
   `settlement_days`.
5. `latest_service_options` keeps the latest observation for each provider,
   payment instrument, and receiving method.
6. `add_objective_losses` computes lower-is-better fee, FX-spread,
   settlement-time, and sourced-risk losses. Coverage and access-point losses
   stay inactive unless those fields are present and non-empty in the CSV.
7. `apply_policy_constraints` removes services that violate configured cost,
   settlement-time, transparency, coverage, or access-point limits. Unsupported
   optional constraints are reported instead of silently applied.
8. `pareto_prune` removes services dominated on the available losses before any
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
   FX spread, settlement time, and sourced risk.

Because the dataset has no explicit consumer-use-case labels, the Random Forest
is pseudo-supervised by K-Means output. This is suitable for exploratory
research, not a production credit or eligibility decision.

## Optimization Model

The default constrained model recommends one service option:

```text
minimize    sum_i weighted_score_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

Research settings can also build batch plans by assigning several same-context
transfers to eligible services while enforcing a maximum provider-concentration
share. Each batch plan is then treated as one candidate row.

The QUBO uses one-hot encoding: `n` candidate rows require `n` qubits, and a
measured bitstring is feasible only when exactly one bit is set.
`MODEL_CANDIDATE_CAP` (equal to `DEFAULT_MAX_QUBITS`, 20) limits the number of
model candidates sent to the QUBO, and therefore the circuit width. See
[qubo_formulation.md](qubo_formulation.md#why-not-a-compact-binary-index-encoding)
for why a logarithmic encoding cannot carry the objective.

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
- End-to-end runtime (wall clock, queue included).
- Compute runtime (queue, network, and submission removed) — the ranking basis.
- Device runtime (quantum circuit execution only).
- Stability.
- Time to solution, on both the wall-clock and compute clocks.
- Inferred ML profile and internal policy weights.
- QUBO validation details.
- QAOA circuit details.

See [qaoa_experiment.md](qaoa_experiment.md#cost--three-clocks) for why cost is
ranked on the compute clock rather than wall clock.
