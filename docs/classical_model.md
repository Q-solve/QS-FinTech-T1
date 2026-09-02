# Validated classical provider/service-selection model

## Scope and data provenance

This model selects one directly observed remittance provider/service alternative. It reads only `data/processed/remittance_east_africa_clean.csv`; it does not read or modify the raw workbook. The initial correctness instance filters `PERIOD = 2025_3Q` and `CORRIDOR = KENTZA`. That filter contains 14 distinct alternatives across 11 unmerged provider names.

An alternative is the exact source tuple

\[
(\text{FIRM},\ \text{FIRM_TYPE},\ \text{PAYMENT INSTRUMENT},\ \text{SPEED ACTUAL},\ \text{PICK-UP METHOD}).
\]

Provider names and comma-delimited method values are preserved exactly. They are neither merged nor split. A stable service-alternative identifier is `alt-` followed by the first 12 hexadecimal characters of the SHA-256 digest of the five UTF-8 source values separated by the unit-separator character. The identifier is deterministic; it is not a claim that similar provider names represent the same entity.

## Benchmarks and eligibility

The model supports only the two amounts quoted by this source period and corridor:

- `CC1`: KES 18,000
- `CC2`: KES 45,000

KES 100,000 is presentation context, not an exact quote in the 2025_3Q data. Every result carries the benchmark label, currency, and amount that supplied its metrics.

Filtering occurs before decision variables are created. An eligible row must have the five alternative fields and finite values for benchmark amount, fee, FX margin, and total-cost percentage. Its benchmark amount must be positive. The filtered rows must map one-to-one to the five-field alternative definition; ambiguous duplicates produce an error rather than an implicit merge.

For validation only, each eligible row must satisfy

\[
\text{TOTAL COST \%} \approx
100\frac{\text{LCU FEE}}{\text{LCU AMOUNT}} + \text{FX MARGIN}
\]

within 0.011 percentage points, the tolerance established by the data audit. Total-cost percentage is returned for display and used in deterministic tie-breaking, but it is not included in the weighted objective because doing so would double count fee and FX margin.

## Objectives and speed encoding

For alternative \(i\), the raw minimization objectives are:

\[
f_i = 100\frac{\text{LCU FEE}_i}{\text{LCU AMOUNT}_i},\qquad
x_i = \text{FX MARGIN}_i,\qquad
s_i = \text{speed ordinal}_i.
\]

The speed labels are ordered as follows:

| Source label | Ordinal |
|---|---:|
| Less than one hour | 0 |
| Same day | 1 |
| Next day | 2 |
| 2 days | 3 |
| 3-5 days | 4 |

These values express order only. They are not hours and must not be interpreted as equal time intervals. The original label is retained in every result.

## Normalization and weights

Each feature is min-max normalized independently within the same eligible comparison set:

\[
\hat z_i = \frac{z_i - \min_j z_j}{\max_j z_j - \min_j z_j}.
\]

Lower normalized values are preferred. If a feature is constant across the comparison set, every normalized value for that feature is zero. This is a neutral contribution and avoids division by zero because the constant feature cannot distinguish alternatives.

Negative fees, margins, or costs are not clamped. A negative minimum therefore maps to zero and all other observations retain their relative position between the observed minimum and maximum.

The user supplies \(w_f\), \(w_x\), and \(w_s\). Each must be finite and in \([0,1]\), and

\[
w_f + w_x + w_s = 1
\]

within an absolute numerical tolerance of \(10^{-9}\). Invalid weights produce a clear validation error.

## Mathematical formulation

Let \(y_i \in \{0,1\}\) indicate whether alternative \(i\) is selected. Define

\[
c_i = w_f\hat f_i + w_x\hat x_i + w_s\hat s_i.
\]

The model is

\[
\min_y \sum_i c_i y_i
\]

subject to

\[
\sum_i y_i = 1,\qquad y_i\in\{0,1\}.
\]

The minimum weighted score is preferred. This one-of-N problem is classically trivial: direct argmin is linear in the number of alternatives once scores are available.

## Deterministic ties

Alternatives are ordered by:

1. lower weighted score;
2. lower source total-cost percentage;
3. faster speed ordinal;
4. provider name using Python's deterministic string ordering;
5. stable service-alternative identifier.

Scores within \(10^{-12}\) are reported as tied at the winning objective value. The remaining fields select a deterministic recommendation and the result exposes whether tie-breaking was applied and all alternative identifiers tied on weighted score.

## Classical algorithms

Two independently callable exact methods share the same filtering, normalization, objective, and tie rules:

- `direct_weighted_argmin` scores every eligible alternative and returns the first deterministic rank.
- `exhaustive_exactly_one_binary` enumerates all \(2^N\) binary assignments, rejects assignments whose bit sum is not one, and selects the best feasible assignment.

Both return the selected alternative, source benchmark, raw and normalized objective components, total cost, weights, score, solver name, monotonic high-resolution runtime, eligible count, tie information, and complete ranking. Solver agreement checks both the stable alternative identifier and objective value. Timings on the 14-alternative instance are instrumentation only and do not support performance or scaling conclusions.

## Reproducible command

From the repository root, run:

```bash
.venv/bin/python -m backend.scripts.rank_classical \
  --benchmark CC2 \
  --fee-weight 0.4 \
  --fx-weight 0.3 \
  --speed-weight 0.3
```

Use `--benchmark CC1` for KES 18,000. The command prints both solver results, every ranked alternative, and an explicit agreement check.

## Limitations

The records describe direct corridor quotes, not composable routes. This implementation does not perform network routing and introduces no unsupported edges, capacities, or intermediary transfers. It does not interpolate or extrapolate to KES 100,000. Ordinal speed normalization preserves ordering but not real duration ratios. Min-max rankings depend on the eligible comparison set and observed extrema. Negative margins and costs may reflect favorable quotes, reference-rate timing, or data issues; preserving them is not a validation of their economic cause. A single 14-alternative run cannot establish a meaningful runtime comparison or quantum advantage. No QUBO, QAOA, API recommendation endpoint, or frontend integration is included.
