# Classical provider/service-selection model

This document specifies the classical model that QKash uses before building the verified
QUBO.

## Scope

QKash selects an eligible direct remittance option for a source country, destination country, and
transfer amount. The dataset supports provider/service selection; it does not contain route
sequences, intermediate legs, capacities, or edge compatibility for true network routing.

In normal mode one candidate row is one provider/payment/receiving-method service. In batch research
mode one candidate row is a feasible plan assigning several same-context transfers to services.

## Pipeline order

The order is fixed and covered by tests:

1. `load_dataset` reads the CSV once through Streamlit cache.
2. `filter_dataset` filters source and destination.
3. `match_transfer_amount` selects the closest RPW benchmark tier.
4. `prepare_candidates` builds amount-specific fee, FX, total-cost, speed, settlement-day, coverage,
   and transparency features.
5. `latest_service_options` keeps the latest observation per provider/payment/receiving method.
6. `add_objective_losses` computes lower-is-better loss columns.
7. `apply_policy_constraints` applies hard eligibility constraints.
8. `pareto_prune` removes dominated rows.
9. `infer_use_case_policy` runs K-Means profiling and Random Forest policy inference.
10. `score_candidates` computes the final weighted score.
11. `select_qubo_candidates` caps the model set.
12. `build_batch_plans` optionally creates provider-constrained multi-transfer plans.

Filtering and hard eligibility constraints happen before Pareto pruning because dominance is only
meaningful inside the eligible comparison set.

## Eligibility

Rows must have non-null `firm`, `amount_lcu`, `fee_lcu`, `fx_margin`, and `total_cost_pct` for the
selected tier. `firm` must be non-empty. The loader does not silently discard negative FX margins,
zero fees, or known CC2 inconsistencies; those remain source facts.

Hard policy constraints can then require:

- `total_cost_pct <= max_total_cost_pct`;
- `settlement_days <= max_settlement_days`;
- transparent pricing when `require_transparency` is true;
- `coverage_score >= min_coverage_score` when coverage is sourced;
- membership in allowed access points when access point is sourced.

The current audited export does not include `receiving network coverage` or `access point`.
`apply_policy_constraints` reports those optional constraints as unsupported instead of filtering on
empty compatibility columns.

## Objectives

The weighted objective minimizes lower-is-better losses:

| Loss | Source |
| --- | --- |
| `transaction_fee_loss` | Min-max normalized `fee_lcu`. |
| `fx_spread_loss` | Min-max normalized `fx_margin`. |
| `time_loss` | `1 - speed_score`, where faster speed labels score higher. |
| `risk_loss` | Average of sourced transparency, coverage, and access-point losses. |

If a risk field is absent or empty, that component is inactive. For the current audited export,
transparency contributes to risk; coverage and access point remain inactive until added to the data.

`total_cost_pct` is used as a hard policy limit, not as another weighted objective, because it is
derived from fee and FX margin and would double-count them.

## Internal policy inference

The user does not enter objective weights. `qkash/profiling.py` first clusters the eligible services
with K-Means, assigns semantic service-profile labels, then trains a Random Forest classifier on
those cluster-derived labels. The inferred profile selects internal weights over transaction fee,
FX spread, settlement time, and sourced risk.

These labels are pseudo-supervised because the dataset has no true consumer use-case labels. They
are useful for exploratory research, not for regulated eligibility decisions.

## Mathematical selection

For `n` model candidates with weighted scores `s_i`:

```text
minimize    sum_i s_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

`solve_original_exact` solves this by direct minimum and returns every co-optimal index. This is
O(n), exact, and expected to beat QAOA on small single-transfer instances.

## Classical algorithms

| Algorithm | Function | Role |
| --- | --- | --- |
| Business as usual | `business_as_usual_baseline` | Non-optimization control. Picks the most frequently observed firm, then that firm's best-scored row. |
| Exact mathematical baseline | `exact_mathematical_baseline` | Ground truth wrapper around `solve_original_exact`. |
| Lowest-fee heuristic | `lowest_fee_heuristic` | Picks the lowest transaction fee, then weighted score as tie-breaker. |
| Fastest-transfer heuristic | `fastest_transfer_heuristic` | Picks the lowest time loss, then weighted score as tie-breaker. |
| Lowest-FX heuristic | `lowest_fx_heuristic` | Picks the lowest FX-spread loss, then weighted score as tie-breaker. |
| Simulated annealing | `simulated_annealing` | Stochastic QUBO sampler using Metropolis single-bit flips over the one-hot model. |

All solvers return the same result shape: `algorithm`, `best_index`, `samples`, and `runtime_s`.
`validate_solver_outputs` checks those samples before metrics are reported.

## Limitations

- One-of-N selection is classically trivial. The research value comes from larger batch instances,
  policy constraints, provider concentration limits, repeated seeds, and runtime accounting.
- Current batch mode repeats the same source/destination/amount request several times; it is a
  research stressor, not a full production treasury scheduler.
- Coverage and access-point constraints become meaningful only after those fields are added to the
  dataset with real values.
