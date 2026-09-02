# World Bank RPW East Africa dataset audit

This document records what the QKash input data actually contains, which fields the application
reads, and which defects a reader must account for before trusting a result. Every count in this
document was measured against the file described below.

## Executive finding

The dataset supports direct-corridor provider/service selection for East African corridors. It does
not support network routing. The data itself is clean: zero exact duplicates, a fully parseable date
column, and a CC1 cost identity that holds on every row. The open defects are in what the schema
*omits* — two fields QKash reads do not exist in this export — and in one filter bug in
`qkash/data.py`. All are listed under [Data-quality risk summary](#data-quality-risk-summary).

## Source, method, and integrity

QKash reads a single audited file, and it is not tracked by Git. The loader prefers
`data/processed/remittance_east_africa_clean.csv` when present, and otherwise falls back to the
root-level file used in this workspace.

| Property | Value |
| --- | --- |
| Path | `data/remittance_east_africa_clean.csv` |
| Rows | 4,026 |
| Columns | 30 |
| SHA-256 | `0b8f69c8c516fa9ea38951c80724f701ff2b16f2064e37ec3cc66c3b243fac90` |

This is a derived file. It descends from the World Bank Remittance Prices Worldwide workbook for
East African corridors, whose authoritative logical range is columns A:AD over 4,504 records. That
workbook is physically malformed from row 972 onward — each 30-field record repeats horizontally
toward XFD — so only A:AD is the real record. The cleaning step that produced this file resolved the
repetition and removed duplicate records, taking 4,504 logical records to 4,026 rows.

Verify integrity before any data work:

```bash
sha256sum data/processed/remittance_east_africa_clean.csv
```

`qkash.data.DEFAULT_DATA_PATH` resolves to the processed path if it exists, else to this root-level
path. `app.py` calls `Path(...).stat()` before anything else, so a missing file raises
`FileNotFoundError` at startup rather than degrading.

## Schema and loader behaviour

`load_dataset` returns **4,026 rows and 37 columns**: the 30 source columns, two columns QKash
can read but this export does not contain, three source-presence flag columns, and two derived
columns (`date_parsed`, `period_order`).

`ensure_supported_columns` lowercases and strips every label, maps a small alias set through
`COLUMN_ALIASES`, and creates any missing `TEXT_COLUMNS` as empty strings. The ten
`REQUIRED_NUMERIC_COLUMNS` have no such fallback; their absence raises `ValueError`. All ten are
present here.

### Fields QKash reads that this export does not have

| Column | Status | Consequence |
| --- | --- | --- |
| `access point` | Absent | Created empty. `unique_values` returns no options. Not used by the `app.py` sidebar, so no user-visible break; `FilterSpec.access_point` matches nothing if set. |
| `receiving network coverage` | Absent | Created empty. `coverage_score` returns its 0.35 default for **all 4,026 rows**. |

`coverage_score` is kept as a compatibility feature, but the loader also records whether the
coverage column came from the CSV. Hard constraints and risk scoring ignore coverage and access point
when they were created as empty compatibility columns. This must stay true until those fields are
sourced.

### Alias handling

This export names the pickup column `PICK-UP METHOD`, not `pickup method`. `COLUMN_ALIASES` maps
`pick-up method` and `pick_up_method` onto the canonical `pickup method`. Without that entry the
sidebar's pickup filter would have no options and `required_selectbox` would halt the app at
`st.stop()`.

## Reporting periods and dates

54 quarterly periods from `2011_1Q` to `2025_3Q`. `period_order` converts a `YYYY_nQ` label to the
sortable integer `year * 10 + quarter`, and returns `-1` for anything unmatched. **No row yields
`-1`.**

The `date` column is uniformly ISO (`2011-01-24`), spanning 2011-01-24 to 2025-09-03. QKash's
original parser used the fixed format `%d/%b/%Y`, which matches **zero** rows in this file and would
have set `date_parsed` to `NaT` for the entire dataset, silently breaking `latest_per_firm` and the
business-as-usual baseline. `parse_dates` now applies each format in `DATE_FORMATS`
(`%d/%b/%Y`, then `%Y-%m-%d`) to the values still unparsed, so both this export and the older
day/month/year extracts resolve completely. **Measured result: 0 `NaT` values out of 4,026.**

## Countries and directed corridors

Three source countries — Kenya, Rwanda, Tanzania. Five destination countries — Kenya, Rwanda,
South Sudan, Tanzania, Uganda. Eight directed corridors: `KENRWA`, `KENSSD`, `KENTZA`, `KENUGA`,
`RWAKEN`, `TZAKEN`, `TZARWA`, `TZAUGA`.

`corridor` agrees with `source_code` concatenated to `destination_code` on all 4,026 rows.

## Providers and provider types

47 distinct `firm` values across five `firm_type` categories: Bank, Mobile Operator, Money Transfer
Operator, Non-Bank FI, Post office.

Provider names are not deduplicated at source. `Ecobank` and `EcoBank Rapid Transfer` are separate,
as are `United Bank for Africa (UBA)` and `UBA Africash`. These may or may not be the same
commercial entity. QKash does not merge them and must not without a documented, approved mapping.

## Duplicate assessment

**Zero exactly duplicated rows.** Deduplication was performed upstream when this file was derived,
which is the main difference from the unaudited extract.

Near-duplicates remain, legitimately: one provider/period/corridor combination appears once per
pickup-method variant. These are genuine distinct service alternatives, but they mean a Pareto
frontier can contain several rows for one provider.

## Completeness

Only eight columns contain nulls, and two of those are free-text notes:

| Column | Nulls | Share |
| --- | --- | --- |
| `NOTE2` | 3,464 | 86.0% |
| `NOTE1` | 2,276 | 56.5% |
| `pickup method` | 231 | 5.7% |
| `payment instrument` | 24 | 0.6% |
| `cc2 lcu fee` | 4 | 0.1% |
| `cc2 lcu fx rate` | 4 | 0.1% |
| `cc2 fx margin` | 4 | 0.1% |
| `cc2 total cost %` | 4 | 0.1% |

The four incomplete CC2 records are one set of four rows missing the whole CC2 metric group.
Selecting the `cc2` amount tier drops them in `prepare_candidates`; `cc1` retains them.

`NOTE1` and `NOTE2` are the provenance fields, present on 1,750 and 562 rows. They are the only
place the source explains an anomalous cost — 47 of the 105 rows with a negative CC1 FX margin carry
a note. Consult them before treating a negative margin as an error.

## Categorical fields

### Transfer speed

Five values, and `speed_score` maps all five. No unmapped-value gap.

| Value | Rows | `speed_score` | `time_loss` |
| --- | --- | --- | --- |
| Less than one hour | 1,532 | 1.00 | 0.00 |
| Next day | 1,069 | 0.62 | 0.38 |
| Same day | 712 | 0.82 | 0.18 |
| 2 days | 638 | 0.42 | 0.58 |
| 3-5 days | 75 | 0.18 | 0.82 |

### Transparency

`yes` (3,791), `Yes` (162), `no` (73). `prepare_candidates` lowercases before comparing, so the
casing split is handled correctly and `transparent_score` is reliable.

### Pickup method

Eleven distinct non-null values, 231 nulls. Casing and phrasing are inconsistent, and unlike
`transparent` this field is **not** normalized before comparison. `_filter_exact` matches the raw
string, so the UI presents these as distinct choices:

| Value | Rows | Rows returned when selected |
| --- | --- | --- |
| Cash | 1,692 | 1,692 |
| Bank account | 1,333 | 1,333 |
| Bank Account | 401 | 401 |
| Mobile wallet | 246 | 246 |
| Own/partner bank account | 49 | 49 |
| Own/partner Bank Account | 32 | 32 |
| Bank account (same/partner bank) | 32 | 32 |
| Own/partner Bank account | 4 | 4 |
| Own/Partner bank account | 2 | 2 |
| Cash, Mobile | 2 | — not offered |
| Mobile, Cash | 2 | — not offered |

Three defects follow:

1. **Casing fragmentation.** Selecting `Bank account` silently excludes the 401 `Bank Account` rows.
   The four `Own/partner` spellings scatter 87 rows across four separate filter entries.

2. **A dead filter option.** `unique_values` splits every value on commas to build the dropdown, so
   `Mobile, Cash` contributes a standalone `Mobile` option. `_filter_exact` then compares against
   the *whole* raw string, and no row equals `Mobile`. **Selecting `Mobile` always returns zero
   rows**, although 250 rows contain that word. Reproducible:

   ```python
   from qkash.data import load_dataset, unique_values, filter_dataset, FilterSpec
   raw = load_dataset()
   assert "Mobile" in unique_values(raw, "pickup method")
   assert len(filter_dataset(raw, FilterSpec(pickup_method="Mobile"))) == 0
   ```

   The mismatch is that `unique_values` tokenizes while `_filter_exact` does not. `payment
   instrument` is routed through `_filter_token` instead and does not suffer this; `pickup method`
   is not.

3. **Under-counting.** For the same reason, selecting `Cash` returns 1,692 rows and misses the four
   `Cash, Mobile` / `Mobile, Cash` rows that also offer cash pickup.

### Payment instrument

16 distinct non-null combinations, only 24 nulls. This is the field most improved by the audited
cleaning — the unaudited extract left 969 nulls here, 21.5% of its rows.

It is a comma-separated multi-value field filtered with `_filter_token`, which splits the stored
value and matches a single selected token against the resulting set, so `Bank account transfer,Cash`
correctly matches a `Cash` selection.

## Cost consistency and validity

The expected identity is `total cost % = fee / amount * 100 + fx margin`.

- **CC1 holds on all 4,026 rows.** Maximum absolute deviation 0.01 percentage points — rounding.
- **CC2 fails on 19 rows.** Maximum absolute deviation 6.21 percentage points. These are the known
  source inconsistencies, retained rather than dropped.

Negative and zero values are present and are legitimate source data:

| Column | Minimum | Negatives | Zeros |
| --- | --- | --- | --- |
| `cc1 lcu fee` | 0.00 | 0 | 80 |
| `cc1 fx margin` | -13.08 | 105 | 88 |
| `cc1 total cost %` | -9.00 | 8 | 3 |
| `cc2 lcu fee` | 0.00 | 0 | 50 |
| `cc2 fx margin` | -13.08 | 105 | 88 |
| `cc2 total cost %` | -8.72 | 9 | 3 |

`NOTE1` attributes negative costs to promotions and special offers. Because `_minmax_loss` maps the
column minimum to 0.0, a promotional negative FX margin becomes the best possible FX score. That is
arithmetically correct and commercially misleading: a promotion is not a durable price. Flag it
rather than removing it.

## Kenya → Tanzania reference slice

Filtering `period = 2025_3Q`, `source_code = KEN`, `destination_code = TZA` yields **14 rows across
11 distinct firms** — the exact-unique 2025_3Q alternatives for that corridor. The unaudited extract
returned 26 rows for the same filter; the difference is duplicate removal.

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
| 1 | `receiving network coverage` absent, so `coverage_score` returns 0.35 for all 4,026 rows | High once coverage becomes an objective | Guarded by source-presence flags |
| 2 | `pickup method` option `Mobile` is offered by the UI but always returns zero rows, because `unique_values` tokenizes while `_filter_exact` does not | High | Open |
| 3 | `pickup method` casing variants fragment the filter and silently exclude rows | Medium | Open |
| 4 | `access point` absent, so that filter dimension is unavailable | Medium | Guarded by source-presence flags |
| 5 | 19 CC2 cost-identity failures, up to 6.21pp | Medium | Retained and documented |
| 6 | Negative FX margins normalize to the best possible score | Medium | Retained; flag in UI |
| 7 | 231 null `pickup method` values (5.7%) are excluded by any pickup filter | Low | Inherent to source |
| 8 | Four rows missing the whole CC2 metric group | Low | Dropped on `cc2` tier |
| 9 | `Ecobank` / `EcoBank Rapid Transfer` and UBA name splits | Low | Deliberately not merged |

Two risks carried by the previous unaudited extract are **resolved** by this file: exact duplicate
rows (now zero) and the 969 null `payment instrument` values (now 24). A third — 246 unparseable
2025 dates — is resolved in code by `parse_dates`.

## Obtaining the data

The file is excluded from Git by `data/**/*.csv`. Obtain
`remittance_east_africa_clean.csv` from the project owner, place it at either
`data/processed/remittance_east_africa_clean.csv` or `data/remittance_east_africa_clean.csv`, and
verify its SHA-256 against
[Source, method, and integrity](#source-method-and-integrity) before use.

The test suite builds its own synthetic DataFrame and does not read this file, so `pytest` passes on
a clean clone with no data present.
