# World Bank RPW East Africa dataset audit

This document records what the QKash input data actually contains, which fields the application
reads, and which defects a reader must account for before trusting a result. Every count in this
document was measured against the files described below.

## Executive finding

The dataset supports direct-corridor provider/service selection for East African remittance
corridors. It does not support network routing. Three defects materially affect current QKash output
and are unresolved: 246 rows in the two 2025 periods fail date parsing, the `receiving network
coverage` field uses two incompatible vocabularies, and one `pickup method` filter option can never
match a row. All are described under [Data-quality risk summary](#data-quality-risk-summary).

## Source, method, and integrity

Two CSV files are used. Neither is tracked by Git; both must be placed in `data/` before `app.py`
will start.

| File | Rows | Columns | SHA-256 |
| --- | --- | --- | --- |
| `data/rpw_dataset_2011_2025_q3_EastAfrica.csv` | 4,504 | 39 | `d6a657c5f0068057c36fc682dea054b543d47d907e35246fcdb8364fd3bed16f` |
| `data/rpw_dataset_2011_2025_q3_EastAfrica2.csv` | 4,504 | 30 | `ba20293a945ab56b0373064e8c6f379b2efc87bdde2ad88afac3ab5610ca9196` |

The first file is the only one the application reads. `qkash.data.DEFAULT_DATA_PATH` resolves to it,
and `app.py` calls `Path(DEFAULT_DATA_PATH).stat()` before anything else, so a missing file raises
`FileNotFoundError` at startup rather than degrading.

The second file is the 30-column source export retained for provenance. It carries a blank spacer
row as its first record (4,505 physical rows, 4,504 logical records) and two free-text columns,
`NOTE1` and `NOTE2`, that the 39-column file drops. Those notes are the only place where the source
explains a negative cost, so consult them before treating a negative FX margin as an error.

Verify integrity before any data work:

```bash
sha256sum data/rpw_dataset_2011_2025_q3_EastAfrica.csv
```

## Workbook shape

`load_dataset` returns 4,504 rows and 41 columns: the 39 CSV columns plus two derived columns,
`date_parsed` and `period_order`. `ensure_supported_columns` lowercases and strips every label, maps
five underscore aliases through `COLUMN_ALIASES`, and creates any of the 12 `TEXT_COLUMNS` that are
absent as empty strings. Ten `REQUIRED_NUMERIC_COLUMNS` have no such fallback; their absence raises
`ValueError`.

## Reporting periods and dates

The data spans 54 quarterly periods from `2011_1Q` to `2025_3Q`. `period_order` converts a
`YYYY_nQ` label to the sortable integer `year * 10 + quarter`, and returns `-1` for anything that
does not match, including empty strings.

`load_dataset` parses the `date` column with the fixed format `%d/%b/%Y` and `errors="coerce"`.
**246 rows fail this parse and receive `NaT`.** Every failing row lies in `2025_1Q` or `2025_3Q`,
because those two periods store ISO timestamps (`2025-02-18 00:00:00`) rather than the
`24/Jan/2011` form used by the other 52 periods. No `date` value is null; the failure is purely a
format mismatch.

The consequence is not cosmetic. `date_parsed` is a sort key in two places:

- `prepare_candidates(latest_per_firm=True)` sorts by `firm` then `date_parsed` descending and keeps
  the first row per firm. `NaT` sorts last, so enabling this option systematically discards the 2025
  observations in favour of older ones.
- `business_as_usual_baseline` sorts by market presence, then `date_parsed` descending, then score.

Any analysis that relies on recency is affected until the parser accepts both formats.

## Countries and directed corridors

Three source countries — Kenya, Rwanda, Tanzania. Five destination countries — Kenya, Rwanda,
South Sudan, Tanzania, Uganda. Eight directed corridors: `KENRWA`, `KENSSD`, `KENTZA`, `KENUGA`,
`RWAKEN`, `TZAKEN`, `TZARWA`, `TZAUGA`.

The `corridor` column agrees with `source_code` concatenated to `destination_code` on all 4,504 rows.
No corridor inconsistency exists in this file.

## Providers and provider types

47 distinct `firm` values across five `firm_type` categories: Bank, Mobile Operator, Money Transfer
Operator, Non-Bank FI, Post office.

Provider names are not deduplicated at source. `Ecobank` and `EcoBank Rapid Transfer` are separate
rows, as are `United Bank for Africa (UBA)` and `UBA Africash`. These may or may not be the same
commercial entity. QKash does not merge them, and must not merge them without a documented and
approved mapping.

## Completeness

Only six columns contain nulls:

| Column | Nulls | Share |
| --- | --- | --- |
| `payment instrument` | 969 | 21.5% |
| `pickup method` | 231 | 5.1% |
| `cc2 lcu fee` | 4 | 0.1% |
| `cc2 lcu fx rate` | 4 | 0.1% |
| `cc2 fx margin` | 4 | 0.1% |
| `cc2 total cost %` | 4 | 0.1% |

The four incomplete CC2 records are a single set of four rows missing the whole CC2 metric group.
Selecting the `cc2` amount tier drops them in `prepare_candidates`; selecting `cc1` retains them.

## Categorical fields

### Transfer speed

Five values, and `speed_score` maps all five. This field has no unmapped-value gap.

| Value | Rows | `speed_score` |
| --- | --- | --- |
| Less than one hour | 1,612 | 1.00 |
| Next day | 1,113 | 0.62 |
| 2 days | 889 | 0.42 |
| Same day | 815 | 0.82 |
| 3-5 days | 75 | 0.18 |

### Receiving network coverage

This field mixes two vocabularies that were never reconciled:

| Value | Rows | `coverage_score` |
| --- | --- | --- |
| High | 1,922 | 0.35 (unmapped, default) |
| Medium | 820 | 0.35 (unmapped, default) |
| Low | 793 | 0.35 (unmapped, default) |
| Nationwide | 803 | 1.00 |
| Major cities | 152 | 0.65 |
| Main city | 14 | 0.35 |

`coverage_score` recognizes only the second vocabulary. **3,535 rows — 78% of the dataset — receive
the 0.35 fallback**, which is also the legitimate score for `Main city`. The resulting column cannot
distinguish "unmapped" from "worst coverage".

This defect is currently latent rather than active: `score_candidates` uses only `fee_lcu`,
`speed_score`, and `fx_margin`, so `coverage_score` is computed and never consumed. It must be fixed
before coverage is promoted to an objective.

### Transparency

`yes` (4,185), `Yes` (246), `no` (73). `prepare_candidates` lowercases before comparing, so the
casing split is handled correctly and `transparent_score` is reliable.

### Pickup method

Eleven distinct non-null values. Casing and phrasing are inconsistent, and unlike `transparent` this
field is **not** normalized before comparison. `_filter_exact` matches the raw string, so the UI
presents these as distinct choices:

| Value | Rows | Rows returned when selected |
| --- | --- | --- |
| Cash | 1,779 | 1,779 |
| Bank account | 1,723 | 1,723 |
| Bank Account | 402 | 402 |
| Mobile wallet | 246 | 246 |
| Own/partner bank account | 49 | 49 |
| Own/partner Bank Account | 32 | 32 |
| Bank account (same/partner bank) | 32 | 32 |
| Own/partner Bank account | 4 | 4 |
| Own/Partner bank account | 2 | 2 |
| Cash, Mobile | 2 | — not offered |
| Mobile, Cash | 2 | — not offered |

Three defects follow from this field:

1. **Casing fragmentation.** Selecting `Bank account` silently excludes the 402 `Bank Account` rows.
   The four `Own/partner` spellings scatter 87 rows across four separate filter entries.

2. **A dead filter option.** `unique_values` splits every value on commas to build the dropdown, so
   `Mobile, Cash` contributes a standalone `Mobile` option. `_filter_exact` then compares against
   the *whole* raw string, and no row equals `Mobile`. **Selecting `Mobile` always returns zero
   rows**, although 250 rows contain that word. This is reproducible:

   ```python
   from qkash.data import load_dataset, unique_values, filter_dataset, FilterSpec
   raw = load_dataset()
   assert "Mobile" in unique_values(raw, "pickup method")
   assert len(filter_dataset(raw, FilterSpec(pickup_method="Mobile"))) == 0
   ```

   The mismatch is that `unique_values` tokenizes while `_filter_exact` does not. `payment
   instrument` and `access point` avoid this because they are routed through `_filter_token`
   instead; `pickup method` is not.

3. **Under-counting.** For the same reason, selecting `Cash` returns 1,779 rows and misses the four
   `Cash, Mobile` / `Mobile, Cash` rows that also offer cash pickup.

### Payment instrument and access point

Both are comma-separated multi-value fields, and both are filtered with `_filter_token`, which
splits the stored value and matches a single selected token against the resulting set. A row holding
`Bank account transfer,Cash` therefore matches a `Cash` selection correctly. These two fields do not
suffer the `pickup method` defect above.

`payment instrument` holds 11 distinct non-null combinations; `access point` holds 17.

## Cost consistency and validity

The expected identity is `total cost % = fee / amount * 100 + fx margin`.

- **CC1 holds on all 4,504 rows.** Maximum absolute deviation is 0.01 percentage points, which is
  rounding.
- **CC2 fails on 19 rows.** Maximum absolute deviation is 6.21 percentage points. These are the same
  19 inconsistencies noted in the source workbook, and they are retained rather than dropped.

Negative and zero values are present and are legitimate source data, not errors:

| Column | Minimum | Negatives | Zeros |
| --- | --- | --- | --- |
| `cc1 lcu fee` | 0.00 | 0 | 87 |
| `cc1 fx margin` | -13.08 | 120 | 88 |
| `cc1 total cost %` | -9.00 | 8 | 3 |
| `cc2 lcu fee` | 0.00 | 0 | 51 |
| `cc2 fx margin` | -13.08 | 120 | 88 |
| `cc2 total cost %` | -8.72 | 9 | 3 |

`NOTE1` in the 30-column file attributes negative costs to promotional offers. Because
`_minmax_loss` maps the minimum to 0.0, a promotional negative FX margin becomes the best possible
FX score. That is arithmetically correct and commercially misleading: a promotion is not a durable
price. Flag it rather than removing it.

## Duplicate assessment

Zero exactly duplicated rows across all 39 columns. Ignoring the synthetic `id` column, exactly one
duplicate pair remains.

Near-duplicates are common and expected. A single provider/period/corridor combination appears once
per pickup method or access-point variant; the 2025_3Q Kenya→Tanzania slice holds 26 rows for 11
distinct firms. These are genuine distinct service alternatives, not data errors, but they mean a
Pareto frontier can contain several rows for one provider.

## Kenya → Tanzania reference slice

Filtering `period = 2025_3Q`, `source_code = KEN`, `destination_code = TZA` yields **26 rows across
11 distinct firms**. The slice illustrates three properties documented above: repeated rows per
provider and pickup method, the `Ecobank` / `EcoBank Rapid Transfer` name split, and a negative FX
margin (Western Union, -0.69) alongside a very low total cost of 0.46%.

## Fitness for optimization

### Supported: direct provider/service selection

Every field required to choose one alternative from a filtered set of direct-corridor options is
present: provider identity, fee, FX margin, total cost, delivery speed, pickup method, and payment
instrument, all resolved to a period and corridor.

### Not supported: true network routing

The file contains no route sequences, intermediate transfer legs, per-provider capacities, edge
compatibility, or feasibility constraints. Any routing claim would require data this dataset does
not have. Dijkstra and comparable graph algorithms are relevant only to a genuine graph-routing
extension built on defensible edge data.

## Data-quality risk summary

| # | Risk | Severity | Status |
| --- | --- | --- | --- |
| 1 | 246 rows in `2025_1Q` and `2025_3Q` fail `%d/%b/%Y` parsing and get `NaT`, corrupting every recency sort | High | Open |
| 2 | `coverage_score` leaves 3,535 rows (78%) on the 0.35 default, indistinguishable from `Main city` | High once coverage becomes an objective | Open, currently unused |
| 3 | `pickup method` option `Mobile` is offered by the UI but always returns zero rows, because `unique_values` tokenizes while `_filter_exact` does not | High | Open |
| 4 | `pickup method` casing variants fragment the filter and silently exclude rows | Medium | Open |
| 5 | 969 null `payment instrument` values (21.5%) are excluded by any instrument filter | Medium | Inherent to source |
| 6 | 19 CC2 cost-identity failures, up to 6.21pp | Medium | Retained and documented |
| 7 | Negative FX margins normalize to the best possible score | Medium | Retained; flag in UI |
| 8 | Four rows missing the whole CC2 metric group | Low | Dropped on `cc2` tier |
| 9 | `Ecobank` / `EcoBank Rapid Transfer` and UBA name splits | Low | Deliberately not merged |

## Obtaining the data

The CSVs are excluded from Git by `data/*.csv` in `.gitignore`. They are World Bank Remittance
Prices Worldwide extracts for East African corridors. Obtain them from the project owner or rebuild
them from the RPW source, place both files in `data/`, and verify the SHA-256 values in
[Source, method, and integrity](#source-method-and-integrity) before use.
