# Validated classical provider/service-selection model

This document specifies the objective, normalization, and classical algorithms that QKash uses, and
records a reproducible reference instance. It describes the code in `qkash/data.py`,
`qkash/scoring.py`, and `qkash/classical.py` as implemented.

## Scope and data provenance

The model selects exactly one provider/service alternative from a filtered set of direct-corridor
observations. It is a one-of-N selection over a fixed candidate set. Nothing in this model routes,
splits, or sequences a transfer.

Source data and its defects are documented in [`data_audit.md`](data_audit.md). The two defects that
bear directly on this model are the 246 unparsed 2025 dates and the 78% unmapped
`receiving network coverage` values.

## Pipeline order

The order is fixed and enforced by test:

1. `load_dataset` — read CSV, normalize labels, derive `date_parsed` and `period_order`.
2. `filter_dataset` — apply every user filter.
3. `prepare_candidates` — select the amount tier and build numeric features.
4. `score_candidates` — normalize objectives and compute the weighted score.
5. `select_qubo_candidates` — Pareto prune, then backfill to a fixed model size.
6. Solve.

Filtering precedes pruning because Pareto dominance is only meaningful within an eligible comparison
set. `tests/test_core.py::test_filtering_happens_before_pareto_pruning` fails if the order is
inverted.

## Benchmarks and eligibility

Two benchmark amount tiers exist in the source data and are selected by `amount_tier`:

- `cc1` — 200 denomination units.
- `cc2` — 500 denomination units.

A row is eligible when `firm`, `amount_lcu`, `fee_lcu`, `fx_margin`, and `total_cost_pct` are all
non-null for the selected tier, and `firm` is a non-empty string. `prepare_candidates` drops
everything else. The optional `latest_per_firm` flag additionally keeps only the most recent row per
firm — see the recency warning in the data audit before enabling it.

## Objectives and encoding

Three objectives, all encoded as lower-is-better losses in `[0, 1]`:

| Objective | Source column | Loss |
| --- | --- | --- |
| Transaction fee | `{tier} lcu fee` | `_minmax_loss(fee_lcu)` |
| Transfer time | `speed actual` | `1 - clip(speed_score, 0, 1)` |
| FX spread | `{tier} fx margin` | `_minmax_loss(fx_margin)` |

Transfer time is not a duration. It is the ordinal `speed_score` mapping from the data audit,
inverted so that faster is lower. The five speed bands therefore produce only five distinct
`time_loss` values: 0.00, 0.18, 0.38, 0.58, 0.82.

`total_cost_pct` is deliberately **not** a fourth objective. It is definitionally
`fee / amount * 100 + fx margin`, so including it alongside the fee and FX-spread objectives would
double-count both.

## Normalization and weights

`_minmax_loss` fills nulls with the column median, then maps the column linearly onto `[0, 1]`. When
the maximum equals the minimum it returns all zeros, so a constant objective contributes nothing
rather than dividing by zero.

Normalization is computed **over the whole filtered pool**, before Pareto pruning and before the
model subset is chosen. A loss value is therefore relative to every eligible row in the corridor,
not to the five rows that reach the QUBO. This keeps scores comparable across runs with the same
filter but means the modelled candidates rarely span the full `[0, 1]` range.

Weights come from `parse_weight_query`, then `normalize_weights` clamps negatives to zero and scales
the three weights to sum to one. The parser supports two forms:

- **Keyword intent** — each objective scores one point per matching keyword substring found in the
  query, and its weight becomes `0.05 + hits`.
- **Explicit override** — `fee=0.5 time=0.3 fx=0.2` sets a weight directly by regex.

An empty query, or one whose weights sum to zero, falls back to `DEFAULT_WEIGHTS`
(0.34 / 0.33 / 0.33).

**Known quirk.** Keyword counting is by substring, and the three keyword lists are not the same
length, so a query that reads as balanced is not. The shipped default,
`"low transaction fee, fast transfer time, low FX spread"`, matches two fee keywords, two time
keywords, and three FX keywords, producing weights of **0.2867 / 0.2867 / 0.4266**. FX spread is
weighted 49% higher than the other two objectives out of the box. Use the sidebar sliders or an
explicit override when a specific weighting is intended.

## Pareto pruning and model selection

`pareto_prune` removes any candidate that some other candidate matches or beats on all three loss
dimensions and strictly beats on at least one, with a `1e-12` tolerance.

Strict pruning often leaves fewer than `TARGET_QUBITS` rows. `select_qubo_candidates` then fills the
remaining slots with the next-best scored rows from the already-filtered pool and marks them
`Best scored fallback` in the `model_source` column; frontier rows are marked `Pareto frontier`.
Every qubit still maps to a real observation. `app.py` surfaces the fallback count in the UI and
refuses to run when fewer than `TARGET_QUBITS` real candidates exist.

## Mathematical formulation

Let `C` be the modelled candidate set, `|C| = n`, and `s_i` the weighted score of candidate `i`.

```
minimize    sum_i s_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

`solve_original_exact` solves this by direct minimum: the optimum is `min(s)`, and all indices within
`numpy.isclose` of that minimum are returned as co-optimal. This is O(n) and exact. The QUBO and
QAOA layers exist to study the formulation, not because this problem is hard.

## Deterministic ties

Every sort in the pipeline uses `kind="mergesort"` — a stable sort — so ties resolve by prior order
rather than arbitrarily. `solve_original_exact` returns the full co-optimal index list in
`indices` and the first as `index`. Metric ranking treats a hit on any co-optimal index as a hit.

## Classical algorithms

| Algorithm | Function | Role |
| --- | --- | --- |
| Business as usual | `business_as_usual_baseline` | Non-optimization control. Picks the most frequently observed firm in the filtered set, then that firm's best-scored row. Represents staying with the incumbent. |
| Exact mathematical baseline | `exact_mathematical_baseline` | Ground truth. Wraps `solve_original_exact`. |
| Simulated annealing | `simulated_annealing` | Stochastic QUBO baseline. Metropolis single-bit flips, geometric cooling from `max(1, penalty + max(scores))` to `1e-3`, default 128 reads × 600 sweeps. |

All three return the same result shape — `algorithm`, `best_index`, `samples`, `runtime_s` — so
`summarize_samples` can score them identically against the QAOA output.

## Reference instance

Kenya → Tanzania, pickup method `Cash`, `cc1` tier, default priority query, `TARGET_QUBITS = 5`.

```
filtered rows   292
prepared rows   292
scored rows     292
Pareto frontier   6
model candidates  5
```

| # | Firm | Period | Speed | Fee (LCU) | FX margin | Fee loss | Time loss | FX loss | Score | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Western Union | 2025_3Q | < 1 hour | 207.0 | -0.69 | 0.034052 | 0.0 | 0.115936 | **0.059218** | Pareto frontier |
| 1 | Western Union | 2020_2Q | < 1 hour | 1320.0 | -3.52 | 0.217145 | 0.0 | 0.000000 | 0.062258 | Pareto frontier |
| 2 | M-Pesa | 2020_2Q | < 1 hour | 100.0 | 0.29 | 0.016450 | 0.0 | 0.156084 | 0.071298 | Pareto frontier |
| 3 | MoneyGram | 2019_4Q | < 1 hour | 30.0 | 1.59 | 0.004935 | 0.0 | 0.209340 | 0.090714 | Pareto frontier |
| 4 | M-Pesa | 2021_4Q | < 1 hour | 0.0 | 3.17 | 0.000000 | 0.0 | 0.274068 | 0.116910 | Pareto frontier |

Exact optimum: candidate 0, objective `0.059218477950031836`, unique.

## Limitations

- **The reference instance is not a menu of purchasable options.** No period filter is applied by
  default, so the five candidates span 2019_4Q to 2025_3Q. The model compares a 2025 Western Union
  quote against a 2020 one as if both were available today. Set a period range before drawing any
  commercial conclusion.
- **The time objective does nothing here.** All five candidates are `Less than one hour`, so
  `time_loss` is 0.0 across the model set and the weighted score is decided entirely by fee and FX
  spread. This is common: speed correlates strongly with Pareto optimality.
- **Candidate 1 wins its FX score through a -3.52 promotional margin.** Section
  [Cost consistency and validity](data_audit.md#cost-consistency-and-validity) explains why that is
  not a durable price.
- **The same provider occupies three of five slots.** Western Union and M-Pesa take four. A
  five-qubit model over near-duplicate rows is a smaller decision than its size suggests.
- **One-of-N selection is classically trivial.** `solve_original_exact` answers it in O(n). Any
  research claim must come from a harder formulation — multiple transactions, capacities, budgets,
  or diversification constraints — not from this instance.

## Reproducible command

```bash
python - <<'PY'
from qkash.data import load_dataset, FilterSpec, filter_dataset, prepare_candidates
from qkash.scoring import (
    parse_weight_query, score_candidates, select_qubo_candidates, solve_original_exact,
)

raw = load_dataset()
weights = parse_weight_query("low transaction fee, fast transfer time, low FX spread")
spec = FilterSpec(source_name="Kenya", destination_name="Tanzania", pickup_method="Cash")
scored = score_candidates(prepare_candidates(filter_dataset(raw, spec), amount_tier="cc1"), weights)
model, pruned = select_qubo_candidates(scored, 5)

print(len(pruned), len(model))
print(model[["firm", "period", "weighted_score", "model_source"]].to_string(index=False))
print(solve_original_exact(model["weighted_score"]))
PY
```

Requires only pandas and numpy.
