# QKash handoff — state as of 2026-09-03

Written at the end of a working session, for whoever picks this up next. It records what changed,
what is verified, what is known-broken, and what the next task is. Everything here was measured
against the code in the repository, not assumed.

## Next task

**Presentation of the data and the algorithm.** The docs are now current and safe to present from —
see [Documentation — brought current](#documentation--brought-current). The substantive story they
now tell: QAOA reproduces the one-hot constraint but does **not** discriminate among feasible
candidates at any depth tested (25 seeds, all confidence intervals contain the uniform baseline).
Present that as the honest finding rather than looking for an advantage claim.

## Repository state

- Branch `kelly`, HEAD `0b63c96` "Refactor QUBO encoding to use one-hot representation and enhance
  runtime reporting".
- **Uncommitted**: the runtime-accounting and metric-ranking work (`app.py`, `qkash/benchmark.py`,
  `qkash/quantum.py`, `tests/test_core.py`), the Streamlit watcher fix
  (`.streamlit/config.toml`), and the documentation rewrite (`README.md`, `AGENTS.md`,
  `docs/*.md`).
- 36 tests pass. Run with `./.venv/bin/python -m pytest -q`.
- `.venv` (Python 3.14.4) has pandas 3.0.5, numpy 2.5.2, scipy 1.18.1, qiskit 2.5.2,
  qiskit-aer 0.17.2, streamlit 1.63.0, scikit-learn 1.9.0, qbraid, pytest 9.1.1.
- `app.py` launches and completes end-to-end (verified with `streamlit.testing.v1.AppTest`,
  including a "Run optimization" click). The **qBraid remote path still cannot be exercised** here —
  no API key — so device-timing extraction remains unverified against a live job.

## The critical fix this session: the QUBO was solving a planted answer

`build_selection_qubo` used to call `solve_original_exact` **first**, then construct a Hamiltonian
around the answer it had just computed classically. The resulting energy function was exactly:

```
E(x) = min(scores) + penalty * hamming_distance(x, classically_computed_optimum)
```

Consequences, all measured:

- The QUBO matrix was **independent of every non-optimal candidate score**. Building it from
  `[0.10, 0.50, 0.90, 0.95]` and from `[0.10, 0.11, 0.12, 0.13]` produced byte-identical matrices.
- Energy ordering did not match score ordering. For scores
  `[0.01, 0.02, 0.03, 0.04, 0.90, 0.91, 0.92, 0.93]`, candidate 4 (score 0.90, near-worst) had energy
  2.010 while candidate 3 (score 0.04, near-best) had energy 4.010 — purely because of bit patterns.
- `verify_qubo_equivalence` was a tautology. **0 failures in 5,850 trials** across `n = 1..39` and
  penalties from 0.001 to 50, because it verified the QUBO against the answer used to build it.
- QAOA reported **100% optimum-hit probability, 100% feasibility, zero gap**, all 512 shots on one
  index. That number was meaningless. It measured a Hamming funnel with a single planted minimum.

This violated the repository's own rule in `AGENTS.md`: *"Never display fabricated quantum savings,
advantage, or hardware results."*

### What replaced it

One-hot encoding, one qubit per candidate, representing the objective exactly:

```
H(x) = sum_i s_i * x_i + A * (sum_i x_i - 1)^2
Q[i][i] = s_i - A ,  Q[i][j] = 2A  (i < j) ,  offset = A
```

Every candidate score now enters the Hamiltonian, so energy ordering of feasible states *is* score
ordering. `verify_qubo_equivalence` can genuinely fail (it does when `A <= min(s)`, where the
all-zeros state wins).

### Why not the compact binary-index encoding — the precise reason

An earlier version of this justification was overstated and has been corrected in the
`build_selection_qubo` docstring. The measured facts, for `k = ceil(log2 n)` index bits:

- A QUBO over `k` bits spans only `1 + k + k(k-1)/2` of the `2**k` basis functions.
- The valid-state block is **rank deficient**, because many pairwise monomials vanish on the valid
  set. For `n = 9`: 11 parameters, 9 equations, but effective **rank 8** — the monomials `x0x1`,
  `x0x2`, `x0x3` are identically zero on all 9 valid states. Overdetermined, so no exact fit.
- Exact fit is impossible for **every `n >= 8`**.
- For `3 <= n <= 7` an exact fit of the valid states exists, but the unused basis states then fall
  **below** the optimum, breaking feasibility.
- Only `n` in `{2, 4}` — where `2**k == n` exactly and there are no unused states — is fully safe.

So "9 candidates should be 4 qubits" is right about *encoding capacity* and wrong about whether a
quadratic Hamiltonian can carry the objective there. The qubit saving is real; the objective is not
representable. A HOBO with ancilla quadratization is the only route to compactness.

## Runtime accounting (uncommitted)

The problem: `summarize_samples` was fed total wall clock, so qBraid was charged for queue,
network, and submission time. Measured on an 8-qubit local run, wall-clock TTS was **52x** the
machine-time TTS *before any network was involved* — the local COBYLA loop was 98% of wall clock.

Three clocks now, each defined deliberately:

| Metric | Meaning |
| --- | --- |
| `end_to_end_runtime_s` | Total cost of using the backend, queue included |
| `compute_runtime_s` | Work the algorithm actually did — **this is the ranking basis** |
| `device_runtime_s` | Circuit execution on the quantum device; `NaN` for classical solvers |

Plus `time_to_solution_s` and `compute_time_to_solution_s`, and `optimizer_runtime_s` broken out.

`METRIC_RANKING_RULE` now ranks on `compute_time_to_solution_s` then `compute_runtime_s`, so a busy
provider queue cannot change solver ordering. `METRIC_RANKING_FALLBACKS` maps each compute column to
its wall-clock equivalent, because NaN-filling a missing cost column silently tied every solver and
left ordering to input order — an existing test caught this.

### qBraid findings, from reading the SDK

- **No seed support.** The native path `QbraidDevice.run` → `submit` → `JobRequest` has no seed
  field. Every `seed` in the SDK is in visualization, random-circuit generation, or the *Azure*
  result builder. `runtime_options: dict[str, Any]` reaches `JobRequest(runtimeOptions=...)` so a
  seed could be *sent*, but nothing indicates native devices consume it.
- **Machine time is available.** `TimeStamps.executionDuration` (milliseconds) is measured in
  `qbraid_core/services/quantum/runner.py` with `time.perf_counter()` around the simulation itself,
  excluding queue and network.
- **Caveat**: a `set_execution_duration` validator falls back to `endedAt - createdAt` when the
  device reports nothing, and `createdAt` is job *creation*, so that fallback **includes queue
  time**. `_extract_qbraid_device_runtime` detects this (values agree within 1 ms) and labels it
  `"qBraid endedAt-createdAt (includes queue)"`.
- **Batch submission exists**: `submit(..., as_batch=True)` sends N circuits in one API call, one
  QRN, gated on `device.profile.batch_job_support`. Not implemented — with a fixed circuit and no
  seed there is nothing to vary. Relevant if a seed sweep is ever added.

**Unverified**: `_extract_qbraid_device_runtime` has three unit tests against mock timestamp objects
but has **never run against a live qBraid job**. Field names come from reading qbraid_core 0.12.2.
Confirm with one real run before relying on the numbers.

## Documentation — brought current

All docs were rewritten against the code and every figure in them re-measured:

- `qubo_formulation.md` — full rewrite for one-hot, with the rank-deficiency reason a compact
  binary-index QUBO cannot work, the falsifiable penalty condition, a re-measured reference
  instance and energy spectrum, and a historical note on the removed planted-answer construction.
- `qaoa_experiment.md` — one-hot decoding and validation, the three runtime clocks, qBraid device
  timing and its queue-inclusive fallback, and 25-seed reference results with confidence intervals.
- `algorithm_flow.md`, `classical_model.md`, `development_plan.md`, `README.md` — encoding, metric
  list, and pipeline-step numbering corrected (`README` said "steps 8 and 10 are gates" after the
  steps had been renumbered; the gates are 13 and 15).
- `AGENTS.md` — removed the stale `TARGET_QUBITS` rule (that constant no longer exists), added a
  standing prohibition on reintroducing the compact encoding, the compute-clock ranking rule, the
  seed-count rule, and the Streamlit watcher requirement.

All internal links and heading anchors verified; every reproducible command in the docs re-run.

## Streamlit segfault — diagnosed and fixed

`python -m streamlit run app.py` segfaulted on startup, twice, in `qiskit/_accelerate.abi3.so` on
the `ScriptRunner` thread.

Root cause: `LocalSourcesWatcher.flush_pending_evictions()` pops watched modules out of
`sys.modules` at the start of each script run. Re-importing `qkash.quantum` and then running Qiskit
work on the fresh ScriptRunner thread crashes. Bisected — it needs all three of eviction,
re-import, and a new thread; any two alone are fine. On WSL the poll-based watcher sees spurious
drvfs mtime changes, so it fires with no file edited.

Fix: `fileWatcherType = "none"` in `.streamlit/config.toml`, with a comment explaining why. Cost:
no hot reload, restart after editing `app.py` or `qkash/*.py`.

## Known-open issues

| # | Issue | Severity |
| --- | --- | --- |
| 1 | ~~Docs describe the removed encoding~~ — **resolved**, all rewritten and re-measured | Closed |
| 2 | `pickup method` option `Mobile` is offered by the UI but always returns 0 rows (`unique_values` tokenizes, `_filter_exact` does not). 250 rows contain the word. | High |
| 3 | `data/rpw_dataset_2011_2025_q3.xlsx` (674 KB) is tracked, against the stated data policy — `.gitignore` covers `data/**/*.csv` only | Medium |
| 4 | `data/lu32416013pb4e.tmp` (0 bytes) is tracked — stray temp file | Low |
| 5 | qBraid device-time extraction unverified against a live job | Medium |
| 6 | `receiving network coverage` / `access point` absent from some exports; `coverage_score` then constant | Medium |

## Strategic note: the challenge asks for routing, and this does not route

The brief's stated decision problem is provider selection, which this solves correctly. But its
title, background, and optional-generalization clause are all **routing**, and the background
specifically attacks *"classical shortest-path routing algorithms"* — never implemented, never
compared against.

The corridor graph **does** support routing, contrary to an earlier claim in `data_audit.md`. There
are no route records, but corridor edges compose: **9 two-leg paths exist**, 5 with a direct
alternative. Critically, **4 have no direct corridor at all** — Rwanda→Uganda, Rwanda→Tanzania,
Rwanda→South Sudan, Tanzania→South Sudan — so provider selection cannot answer them in principle.

Sized on 2025_3Q, choosing a route *and* a provider per leg gives **19–37 qubits** with decision
spaces of 84–151, against the current 5–8. That is the regime where QAOA is worth running, and it
supplies the multi-objective shortest-path baseline the background is about. It needs documented
composition rules: fees add in a common currency, FX spreads compound as `(1+s1)(1+s2)-1`,
settlement times add.

The Pareto layer, QUBO verification gate, and benchmark harness all transfer unchanged.

## Verification commands

```bash
./.venv/bin/python -m pytest -q                  # 36 tests
git diff --stat                                  # the uncommitted runtime work
grep -rn "binary-index" docs/ README.md          # stale doc claims
```
