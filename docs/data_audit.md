# World Bank remittance dataset audit

Audit date: 2026-09-01
Project: **Quantum-Assisted Multi-Objective Optimization of Cross-Border Remittance Services**

## Executive finding

The workbook is suitable for a direct-corridor provider/service selection MVP, with important cleaning and interpretation constraints. It does **not** contain the topology or operational constraints needed for true network routing.

The preliminary logical counts were mostly correct: one sheet, 4,504 records, 30 intended fields, 478 redundant exact copies, 683 Kenya→Tanzania records, and eight directed corridors. However, the full-width issue is more serious than formatting: from worksheet rows 972 through 4506, the 30-field logical record is repeatedly serialized across columns AE:XFD. A:AD remains the intended schema and was the only range used for the tabular audit.

## Source, method, and integrity

- Location during the audit: `frontend/data/raw/rpw_dataset_2011_2025_q3_EastAfrica.xlsx`.
- Corrected repository location: `data/raw/rpw_dataset_2011_2025_q3_EastAfrica.xlsx`.
- Workbook size: 213,467,998 bytes.
- Raw SHA-256 before and after processing: `7d3b394c0db9a8c4227cccd52bb73e09c026f3f1a74de6befdee8b6477ab1ce4` (unchanged).
- Audit method: Python 3.11.16, openpyxl 3.1.5 in `read_only=True`, `data_only=True`, with `min_col=1`, `max_col=30`; pandas 3.0.5 for profiling.
- Worksheet XML dimension: `A1:XFD4506`; uncompressed worksheet XML size: 2,270,856,782 bytes.
- A streaming XML check found 54,697,805 nonempty cells beyond AD, affecting 3,535 consecutive worksheet rows (972–4506). Sampled affected rows repeat the A:AD sequence in 30-column cycles through XFD. This is physical horizontal duplication/corruption, not merely formatting.
- Only the first logical 30 columns were analyzed. The repeated horizontal copies were not interpreted as additional records or fields.

Repository correction on 2026-09-01 restored the approved 213,467,998-byte workbook at the root data location with the same verified SHA-256. A mismatched 546,366-byte workbook found there during correction was quarantined outside the repository rather than treated as the raw source.

## Workbook shape

| Item | Verified result |
|---|---:|
| Sheet names | `Dataset (up to Q1 2016)` |
| Worksheet rows | 4,506 |
| Header row | 1 |
| Blank spacer row | 2 |
| Logical record rows | 4,504 |
| Intended populated fields | 30 (A:AD) |
| Physical worksheet columns | 16,384 (A:XFD) |
| Logical grain | One observed provider/service quote for one directed corridor, reporting period, and two benchmark send amounts |

The sheet name is stale: records continue through `2025_3Q`.

## Exact column names

1. `PERIOD`
2. `SOURCE_CODE`
3. `SOURCE_NAME`
4. `DESTINATION_CODE`
5. `DESTINATION_NAME`
6. `FIRM`
7. `FIRM_TYPE`
8. `PAYMENT INSTRUMENT`
9. `SPEED ACTUAL`
10. `CC1 LCU AMOUNT`
11. `CC1 DENOMINATION AMOUNT`
12. `CC1 LCU CODE`
13. `CC1 LCU FEE`
14. `CC1 LCU FX RATE`
15. `CC1 FX MARGIN`
16. `CC1 TOTAL COST %`
17. `CC2 LCU AMOUNT`
18. `CC2 DENOMINATION AMOUNT`
19. `CC2 LCU CODE`
20. `CC2 LCU FEE`
21. `CC2 LCU FX RATE`
22. `CC2 FX MARGIN`
23. `CC2 TOTAL COST %`
24. `INTER LCU BANK FX`
25. `TRANSPARENT`
26. `NOTE1`
27. `NOTE2`
28. `PICK-UP METHOD`
29. `DATE`
30. `CORRIDOR`

## Reporting periods and dates

There are 54 distinct reporting-period labels, from `2011_1Q` through `2025_3Q`:

`2011_1Q`, `2011_3Q`, `2012_1Q`, `2012_3Q`, `2013_1Q`, `2013_2Q`, `2013_3Q`, `2013_4Q`, `2014_1Q`, `2014_2Q`, `2014_3Q`, `2014_4Q`, `2015_1Q`, `2015_2Q`, `2015_3Q`, `2015_4Q`, `2016_1Q`, `2016_2Q`, `2016_3Q`, `2016_4Q`, `2017_1Q`, `2017_2Q`, `2017_3Q`, `2017_4Q`, `2018_1Q`, `2018_2Q`, `2018_3Q`, `2018_4Q`, `2019_1Q`, `2019_2Q`, `2019_3Q`, `2019_4Q`, `2020_1Q`, `2020_2Q`, `2020_3Q`, `2020_4Q`, `2021_1Q`, `2021_2Q`, `2021_3Q`, `2021_4Q`, `2022_1Q`, `2022_2Q`, `2022_3Q`, `2022_4Q`, `2023_1Q`, `2023_2Q`, `2023_3Q`, `2023_4Q`, `2024_1Q`, `2024_2Q`, `2024_3Q`, `2024_4Q`, `2025_1Q`, `2025_3Q`.

Observed dates range from 2011-01-24 to 2025-09-03. The absence of some quarter labels, especially in the early semiannual portion and `2025_2Q`, is reported as source coverage rather than filled or inferred.

## Completeness

Twenty-three of the 30 columns have no true nulls. The seven columns with nulls are:

| Column | Missing | Rate |
|---|---:|---:|
| `CC2 LCU FEE` | 4 | 0.089% |
| `CC2 LCU FX RATE` | 4 | 0.089% |
| `CC2 FX MARGIN` | 4 | 0.089% |
| `CC2 TOTAL COST %` | 4 | 0.089% |
| `NOTE1` | 2,752 | 61.101% |
| `NOTE2` | 3,942 | 87.522% |
| `PICK-UP METHOD` | 231 | 5.129% |

The four records missing all CC2 service metrics are `Express Money`, Rwanda→Kenya, cash→mobile wallet, in `2016_3Q`, `2016_4Q`, `2017_1Q`, and `2017_2Q`.

`PAYMENT INSTRUMENT` has no true nulls, but contains 24 literal `N/A` values. These remain source categories and were not converted to nulls.

## Duplicate assessment

Exact comparison used all 30 logical fields.

| Measure | Count |
|---|---:|
| Raw logical records | 4,504 |
| Rows participating in an exact-duplicate group | 809 |
| Exact-duplicate groups | 331 |
| Redundant copies after keeping the first occurrence | 478 |
| Exact-unique records | 4,026 |

Trimming surrounding whitespace, collapsing internal whitespace, and case-folding strings did not produce additional whole-row duplicates: normalized counts remained 809 participating rows and 478 redundant copies. Duplicate participation is heavily concentrated from `2022_4Q` onward; `2025_3Q` contains 73 rows participating in duplicate groups across all corridors.

## Countries and directed corridors

Observed sending countries are Kenya (`KEN`), Rwanda (`RWA`), and Tanzania (`TZA`). Observed receiving countries are Kenya (`KEN`), Rwanda (`RWA`), South Sudan (`SSD`), Tanzania (`TZA`), and Uganda (`UGA`). These are all East African countries; no other countries occur in the audited workbook.

| Corridor | Direction | Records |
|---|---|---:|
| `KENRWA` | Kenya → Rwanda | 469 |
| `KENSSD` | Kenya → South Sudan | 552 |
| `KENTZA` | Kenya → Tanzania | 683 |
| `KENUGA` | Kenya → Uganda | 867 |
| `RWAKEN` | Rwanda → Kenya | 330 |
| `TZAKEN` | Tanzania → Kenya | 510 |
| `TZARWA` | Tanzania → Rwanda | 524 |
| `TZAUGA` | Tanzania → Uganda | 569 |

The workbook contains no mention of `Zanzibar` in any of the 30 fields. Treating Zanzibar as Tanzania is therefore a demonstration mapping, not a source-data category.

## Provider, service, cost, and delivery fields

- Provider identity/type: `FIRM`, `FIRM_TYPE`.
- Service dimensions: `PAYMENT INSTRUMENT`, `SPEED ACTUAL`, `PICK-UP METHOD`; `NOTE1` and `NOTE2` may add conditions.
- Fee fields: `CC1 LCU FEE`, `CC2 LCU FEE`.
- Total-cost fields: `CC1 TOTAL COST %`, `CC2 TOTAL COST %`.
- FX-margin fields: `CC1 FX MARGIN`, `CC2 FX MARGIN`.
- Quoted and reference FX fields: `CC1 LCU FX RATE`, `CC2 LCU FX RATE`, `INTER LCU BANK FX`.
- Benchmark context: `CC1/CC2 LCU AMOUNT`, `CC1/CC2 DENOMINATION AMOUNT`, and `CC1/CC2 LCU CODE`.

Provider types and counts are Bank (2,418), Money Transfer Operator (1,564), Mobile Operator (261), Post office (136), and Non-Bank FI (125).

### Transfer speed values

| Value | Records |
|---|---:|
| Less than one hour | 1,612 |
| Next day | 1,113 |
| 2 days | 889 |
| Same day | 815 |
| 3-5 days | 75 |

These are ordered service bands, not measured continuous durations. A documented encoding is required before normalization.

### Payment instruments

`Bank account transfer` (1,826), `Cash` (1,259), `Account to account` (529), `Cash to cash` (336), `Mobile money` (267), `Bank account transfer,Cash` (94), `Cash,Debit/credit card` (40), `Account to account (other bank)` (39), `Account to account (same bank)` (37), `N/A` (24), `Debit card` (18), `Credit Card` (17), `Credit Card,Debit card` (6), `Mobile` (4), `Debit/credit card,Mobile money` (3), `Credit Card,Debit card,Mobile money` (3), and `Debit/credit card` (2).

Comma-delimited values represent multiple available instruments in one source cell; they should not be split into new alternatives without a documented modeling rule.

### Receiving/pick-up methods

`Cash` (1,779), `Bank account` (1,723), `Bank Account` (402), `Mobile wallet` (246), null (231), `Own/partner bank account` (49), `Own/partner Bank Account` (32), `Bank account (same/partner bank)` (32), `Own/partner Bank account` (4), `Own/Partner bank account` (2), `Mobile, Cash` (2), and `Cash, Mobile` (2).

Case variants are present. They may be standardized for categorical analysis with an explicit mapping, while retaining the raw column.

## Cost consistency and validity risks

For CC1, all 4,504 rows satisfy, within 0.011 percentage points:

`CC1 TOTAL COST % = (CC1 LCU FEE / CC1 LCU AMOUNT × 100) + CC1 FX MARGIN`.

This proves that adding `CC1 FX MARGIN` to `CC1 TOTAL COST %` would double-count the margin. For CC2, 4,481 of 4,500 comparable rows satisfy the same identity within 0.011 percentage points. Nineteen rows do not, with absolute residuals up to 6.206 percentage points; they occur in Tanzania→Uganda (2014_4Q–2015_4Q), Rwanda→Kenya (`2017_3Q`), and Kenya→Uganda (`2019_1Q`). These rows require exclusion, correction from an authoritative source, or a sensitivity analysis before CC2-based modeling.

Observed values include negative FX margins (minimum -13.08%) and negative total costs (CC1 minimum -9.00%; CC2 minimum -8.72%). These were not altered. They may reflect favorable quoted FX rates, timing/reference-rate effects, or data issues; do not clamp them without a defensible rule.

## Kenya → Tanzania findings

There are 683 raw Kenya→Tanzania records covering 39 reporting periods from `2015_4Q` through `2025_3Q`. The most recent period contains 26 raw rows and 14 exact-unique logical rows.

For `2025_3Q`, the source benchmarks are KES 18,000 (`CC1`, denomination amount 200) and KES 45,000 (`CC2`, denomination amount 500). The demonstration amount KES 100,000 is not directly observed. A recommendation at KES 100,000 would require an explicit approximation or new quote data; the 45,000-KES percentages must not be presented as an exact 100,000-KES quote.

### Provider-name standardization used for counting

Only surrounding whitespace removal, internal whitespace collapse, and case-folding were tested. This produced the same 11 distinct raw provider names; no provider names were merged. In particular, `Ecobank` and `EcoBank Rapid Transfer` remain separate, as do `United Bank for Africa (UBA)` and `UBA Africash`, because the dataset presents different names/services and no source-backed entity mapping was supplied.

### Genuine 2025_3Q alternatives

An alternative is defined by the exact tuple (`FIRM`, `FIRM_TYPE`, `PAYMENT INSTRUMENT`, `SPEED ACTUAL`, `PICK-UP METHOD`). Under that definition there are **14 distinct alternatives across 11 provider names**:

| Provider | Type | Payment | Speed | Receiving method |
|---|---|---|---|---|
| M-Pesa | Mobile Operator | Mobile money | Less than one hour | Cash |
| M-Pesa | Mobile Operator | Mobile money | Less than one hour | Mobile wallet |
| KCB Bank | Bank | Bank account transfer | 2 days | Bank account |
| United Bank for Africa (UBA) | Bank | Bank account transfer | 2 days | Bank account |
| Ecobank | Bank | Cash | Same day | Cash |
| ABC Bank | Bank | Bank account transfer | 2 days | Bank account |
| UBA Africash | Bank | Cash | Less than one hour | Cash |
| Development Bank of Kenya | Bank | Bank account transfer | 2 days | Bank account |
| Western Union | Money Transfer Operator | Cash | Less than one hour | Cash |
| MoneyGram | Money Transfer Operator | Cash | Less than one hour | Cash |
| MoneyGram | Money Transfer Operator | Cash | Less than one hour | Mobile wallet |
| GT Bank | Bank | Bank account transfer | Next day | Bank account |
| EcoBank Rapid Transfer | Bank | Cash | Less than one hour | Cash |
| EcoBank Rapid Transfer | Bank | Bank account transfer | Less than one hour | Cash |

The two M-Pesa rows and two MoneyGram rows differ by receiving method; the two EcoBank Rapid Transfer rows differ by payment instrument. They are service alternatives even where their observed price metrics are identical.

## Fitness for optimization

### Supported: direct provider/service selection

The data contains a direct corridor, observation period, provider, service method, speed band, two benchmark amounts, fees, FX margins, total-cost percentages, and receiving method. This supports filtering to eligible alternatives and selecting one provider/service according to documented weighted objectives and constraints.

### Not supported by this workbook: true network routing

The eight directed corridor labels can be drawn as a country graph, but the records do not specify multi-leg routes, transfer compatibility between providers, intermediate hand-offs, capacities, settlement links, cumulative timing, per-leg availability, or path feasibility. Treating corridor quotes as composable graph edges would be an unsupported assumption. Dijkstra or route-QUBO comparisons require a separate, validated routing dataset and formulation.

The initial one-of-14 provider choice is classically solvable by direct enumeration/argmin. It is useful as a correctness and demonstration instance, not evidence of quantum advantage. A stronger extension should introduce multiple simultaneous transfers, capacity/budget constraints, provider exposure/diversification, service-level requirements, or validated multi-leg routes.

## Derived dataset and cleaning log

Created `data/processed/remittance_east_africa_clean.csv` with 4,026 records, 30 original fields, size 1,141,351 bytes, and SHA-256 `0b8f69c8c516fa9ea38951c80724f701ff2b16f2064e37ec3cc66c3b243fac90`.

Cleaning operations, in order:

1. Read only A:AD from the raw workbook; ignored the malformed repeated columns AE:XFD.
2. Used row 1 as the 30-field header.
3. Removed blank worksheet row 2.
4. Retained 4,504 nonblank logical records from worksheet rows 3–4506.
5. Removed 478 exact redundant records across all 30 fields, keeping the first occurrence; 4,026 records remain.
6. Converted `DATE` to ISO `YYYY-MM-DD` text.
7. Serialized to UTF-8 CSV with the original 30 column names and order.

No values were imputed. No provider names, country names, service categories, capitalization variants, `N/A` strings, negative values, or CC2 inconsistencies were changed. A reload check returned 4,026 rows, 30 columns, and zero exact duplicate rows.

## Data-quality risk summary

| Finding | Severity | Impact |
|---|---|---|
| Horizontal record repetition through XFD on rows 972–4506 | High | Makes naive workbook loading extremely expensive and could be mistaken for millions of fields/observations. Use A:AD only. |
| 478 redundant exact records | High for experiment design | Inflates provider/corridor counts and can bias sampling or success rates. Deduplicate before defining alternatives. |
| 19 inconsistent CC2 total costs | High for CC2 optimization | Can rank services incorrectly if CC2 total cost is trusted without validation. |
| Demo amount KES 100,000 absent | High for product claims | Prevents an exact source-backed quote for the story amount. |
| Missing CC2 metrics in four rows | Medium | Requires eligibility/exclusion handling for CC2 comparisons. |
| Receiving-method nulls and capitalization variants | Medium | Can split equivalent categories or make service eligibility ambiguous. |
| Negative margins/costs | Medium | May be valid timing/reference effects or anomalies; needs sensitivity checks. |
| Stale worksheet name | Low | Metadata can mislead users about freshness, though data reaches 2025_3Q. |
