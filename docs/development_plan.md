# Staged development plan

## Guiding outcome

Build a reproducible research prototype that recommends an eligible direct remittance provider/service and compares QAOA against appropriate classical baselines without claiming an advantage in advance. The first Kenya→Tanzania instance is a correctness and demonstration case; scaling experiments and stronger constrained formulations carry the research value.

No solver, API, or frontend implementation is part of the completed audit stage.

## Stage 0 — Data foundation (completed audit; pipeline next)

Deliverables:

- Preserve the raw workbook and its SHA-256.
- Treat A:AD as the authoritative schema and make the malformed AE:XFD repetition an automated guardrail.
- Turn the documented one-off cleaning steps into a deterministic, tested data-build command.
- Produce a compact processed table plus a machine-readable audit manifest containing source hash, schema, row counts, duplicate counts, period coverage, and cleaning version.
- Define explicit policies for exact duplicates, four missing CC2 records, 19 CC2 cost-identity failures, `N/A`, category casing, and negative values.
- Keep the corrected raw workbook at root `data/raw/`, excluded from Git, and verify its approved SHA-256 before any raw-data processing.

Exit criteria:

- Rebuilding from the unchanged raw hash produces 4,026 rows and 30 columns.
- Tests confirm the original file is unchanged, output has zero exact duplicates, dates parse, corridor codes agree with source/destination codes, and CC1 cost identity holds within the documented tolerance.

## Stage 1 — Domain contract and MVP eligibility

Define typed domain objects for corridor, observation period, provider, service alternative, benchmark amount, cost components, speed band, payment method, receiving method, and data-quality flags.

For the first instance:

- Filter `PERIOD = 2025_3Q`, `SOURCE_CODE = KEN`, `DESTINATION_CODE = TZA`.
- Map story destination Zanzibar to dataset destination Tanzania in presentation/domain input, while retaining `TZA` in data provenance.
- Deduplicate on all source fields, then identify alternatives by (`FIRM`, `FIRM_TYPE`, `PAYMENT INSTRUMENT`, `SPEED ACTUAL`, `PICK-UP METHOD`). Expected result: 14 alternatives across 11 unmerged provider names.
- Require a benchmark policy. The data offers KES 18,000 and KES 45,000, not KES 100,000. Choose one of: demonstrate with a source-supported amount; label 45,000-KES percentages as an approximation; add an externally sourced fee schedule; or define and validate an interpolation/extrapolation method.
- Define speed ordering (for example, less than one hour < same day < next day < 2 days < 3–5 days) and document the numeric encoding.

Exit criteria:

- A filter report explains every included and excluded row.
- The 14-alternative fixture is derived from data, not hard-coded provider claims.
- All displayed values carry period, corridor, amount benchmark, and source lineage.

## Stage 2 — Objective and normalization specification

Choose a formulation before coding solvers.

Recommended MVP objective:

- One binary decision variable per eligible provider/service alternative.
- Exactly-one constraint: select one alternative.
- Minimize a user-weighted normalized score over non-overlapping objectives.
- Use total-cost percentage and speed for the simplest defensible score. Display fee and FX margin as explanations.
- If fee and FX margin are optimized separately, exclude total cost from that same additive objective or explicitly derive a non-double-counting formulation.

Normalization requirements:

- Apply one documented method consistently within the eligible set, initially min–max normalization to [0,1].
- Record min/max values and whether lower or higher is better.
- Define behavior for constant objectives (neutral zero contribution is preferable to division by zero).
- Do not clamp negative values merely to make plots look intuitive.
- Validate weights (nonnegative and normalized to sum to one, or explain equivalent scaling).
- Add sensitivity analysis over weight grids and report rank stability/Pareto-efficient alternatives.

Exit criteria:

- A written mathematical formulation and test fixtures reproduce hand-calculated scores.
- No objective double counts FX margin.
- The same scoring function is shared by every solver and the API.

## Stage 3 — Classical baselines

Implement classical methods before QAOA:

1. Direct score/argmin baseline for one-of-N selection.
2. Exhaustive enumeration that returns all feasible bitstrings and the exact optimum.
3. Mathematical-optimization baseline using the same binary formulation and constraints.

Record solution, objective value, feasibility, runtime, and optimality certificate/gap. For the one-of-N MVP, direct argmin is the meaningful baseline and should be described as trivial in complexity for the fixed instance.

Exit criteria:

- All classical implementations agree on optimum and ties.
- Tests cover one alternative, ties, constant objective columns, invalid weights, missing values, and infeasible eligibility sets.

## Stage 4 — QUBO/QAOA implementation

- Translate the exact same objective and exactly-one constraint into a QUBO/Ising representation using Qiskit Optimization.
- Verify energies against exhaustive enumeration for every bitstring on small instances.
- Use Qiskit Aer first; fix and record seeds, optimizer, repetitions, depth `p`, shots, transpilation settings, penalty selection, and package versions.
- Separate feasible-sample probability from the probability of sampling the exact optimum.
- Report best sampled solution, expected energy, feasibility rate, success probability, approximation/optimality gap, circuit metrics, optimizer evaluations, and timing components.
- Treat penalty tuning and post-selection as experimental choices and disclose them.

Exit criteria:

- QUBO energy and classical objective equivalence is tested.
- QAOA output is decoded and feasibility-checked without silent repair.
- Results are reproducible across repeated seeded runs.

## Stage 5 — Experiment and scaling framework

Define preregistered experiment families rather than selecting only favorable cases:

- Increasing alternative counts using defensible subsets or synthetic instances whose generation rules are explicit and separate from observed World Bank data.
- Weight-grid sensitivity.
- Noise-free statevector/sampler versus finite-shot simulation, followed later by noisy simulation.
- Multiple random seeds and optimizer initializations.
- QAOA depth and shot-count sweeps.

Compare:

- Solution quality/objective gap to exact optimum.
- Constraint feasibility rate.
- Exact-optimum success probability and top-k probability.
- Runtime: model build, transpilation, quantum evaluation, classical optimizer, sampling, queue/hardware, and end-to-end.
- Scaling in variables, constraints, depth, two-qubit gates, shots, and optimizer evaluations.
- Robustness across weights and seeds.

Use confidence intervals or distributions. Do not claim quantum benefit from simulator runtime or a single successful sample.

## Stage 6 — FastAPI backend

After solver contracts stabilize:

- Add endpoints for metadata/available filters, eligible alternatives, classical recommendation, quantum experiment submission/result, and experiment comparison.
- Use explicit request/response schemas with corridor, period, benchmark, weights, eligibility constraints, solver configuration, provenance, and warnings.
- Keep recommendation endpoints deterministic by default; long quantum experiments should be separated from interactive recommendation.
- Validate unsupported amounts such as KES 100,000 and return a clear approximation/source warning rather than fabricated values.
- Add pytest coverage for schemas, filtering, scoring, solver equivalence, errors, and provenance.

## Stage 7 — React demonstration UI

Replace the existing Vite starter only after the API contract exists.

Suggested flow:

1. Story input: Awatiff, Kenya → Zanzibar (mapped to Tanzania), amount and urgency/preferences.
2. Data-coverage notice showing period and supported benchmark amount.
3. Weight controls and service eligibility filters.
4. Ranked provider/service alternatives with fee, FX margin, total cost, speed, payment, receiving method, and provenance.
5. Classical-versus-QAOA experiment panel showing measured results and uncertainty—not invented savings.
6. Research view with feasibility, optimality gap, success probability, runtime decomposition, and scaling charts.

Use Tailwind, Lucide, and Recharts as planned. Accessibility, responsive behavior, empty/error/loading states, and explicit approximation badges are required.

## Stage 8 — qBraid and hardware execution

- Freeze small validated circuits and experiment manifests before remote execution.
- Record backend, calibration timestamp, queue time, transpiled circuit, shots, mitigation, and all software versions.
- Compare hardware with equivalent noisy/noiseless simulation and exact results.
- Report hardware results as empirical observations with uncertainty, not generalized advantage.

## Stage 9 — Stronger research extension

The one-of-N choice is too simple to establish a meaningful quantum/classical scaling story. Prioritize a multiple-transaction constrained assignment model:

- Assign many remittance requests to provider/service alternatives.
- Add provider or corridor capacity, user deadlines, payment/receiving compatibility, budgets, concentration/diversification limits, and possibly fairness or reliability constraints.
- Optimize total cost, service delay, and constraint penalties across the batch.

This creates interacting binary decisions while staying grounded in provider selection. A true routing extension is separate: acquire validated multi-leg edges, provider interoperability, per-leg costs/speeds, capacities, and path constraints before using Dijkstra, shortest-path baselines, or routing QUBOs.

## Stage gates and claims policy

At every stage, retain a reproducible configuration, source hash, processed-data hash, package versions, random seeds, and raw result artifacts. A result is demo-ready only if classical and quantum methods solve the same documented instance and all metrics are computed from stored outputs. Wording must remain neutral unless repeated experiments support a narrower, qualified conclusion.
