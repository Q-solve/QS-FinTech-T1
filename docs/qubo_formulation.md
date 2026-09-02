# Validated QUBO formulation

This document specifies the QUBO that QKash builds, the penalty condition that makes it equivalent
to the classical constrained model, the Ising conversion used by QAOA, and the verification that
gates every run. It describes `build_selection_qubo`, `verify_qubo_equivalence`, and `qubo_to_ising`
as implemented.

## Source model

After corridor/amount filtering, hard policy constraints, Pareto pruning, ML policy inference, and
optional batch-plan construction, QKash holds `n` model candidate rows with weighted scores `s_i`.
The constrained model is one-of-N selection:

```text
minimize    sum_i s_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

For a single transfer each row is a provider/payment/receiving-method service. In batch research
mode each row is a feasible assignment plan for several transfers.

## One-hot variables

QKash uses **one-hot encoding: one qubit per candidate row**, so `n` candidates require `n` qubits
and `x_i = 1` means candidate `i` is selected. `required_qubits(n)` returns `n`.

This makes the objective *linear* in the decision variables, and therefore representable exactly by
a quadratic form. Every candidate score enters the Hamiltonian, so the energy ordering of feasible
states is the score ordering. That property is what makes a solver comparison meaningful.

### Why not a compact binary-index encoding

Encoding the row number in `k = ceil(log2 n)` bits would use far fewer qubits, and an earlier
version of QKash did exactly that. It was removed because a QUBO over `k` index bits cannot carry
the objective. The reasoning is measured, not asserted:

- A QUBO over `k` bits spans only `1 + k + k(k-1)/2` of the `2**k` basis functions.
- The block of rows for the *valid* indices is rank deficient, because many pairwise monomials
  vanish on that set. For `n = 9`: 11 parameters, 9 equations, but effective **rank 8** — the
  monomials `x0x1`, `x0x2`, `x0x3` are identically zero on all nine valid states. Overdetermined,
  so no exact representation exists.
- Exact representation is impossible for **every `n >= 8`**.
- For `3 <= n <= 7` the valid states can be fitted exactly, but the unused basis states then fall
  *below* the optimum and break feasibility.
- Only `n` in `{2, 4}` — where `2**k == n` and there are no unused states — is fully safe.

So "9 candidates need only 4 qubits" is correct about *encoding capacity* and wrong about whether a
quadratic Hamiltonian can carry the *objective* there. `build_selection_qubo` raises
`NotImplementedError` when asked for `ENCODING_BINARY_INDEX`. A higher-order (HOBO) formulation with
ancilla quadratization is the only route to compactness.

**Historical note.** The removed implementation sidestepped this by solving the model classically
first and building a Hamming-distance well around the answer, giving
`energy(bits) = best_objective + penalty * hamming(bits, best_bits)`. That made the QUBO independent
of every non-optimal score, made `verify_qubo_equivalence` a tautology, and produced QAOA hit rates
of 100% that measured nothing. Do not reintroduce it.

## Constructed QUBO

The one-hot constraint is folded into the objective with a quadratic penalty `A > 0`:

```text
H(x) = sum_i s_i * x_i + A * (sum_i x_i - 1)^2
```

Expanding gives the upper-triangular QUBO that `build_selection_qubo` returns:

```text
Q[i][i] = s_i - A
Q[i][j] = 2A          for i < j
offset  = A
```

`QuboModel.energy` evaluates `offset + sum_{i<=j} Q[i][j] * x_i * x_j`. The constructor rejects an
empty candidate list, any score below `-1e-12`, and a non-positive penalty. With no penalty supplied
it defaults to `max(1, 2 * max(s))`. `app.py` uses `penalty_multiplier * max(1, max(s))`.

## Penalty condition

Because all scores are non-negative, energies partition by Hamming weight:

- **Weight 1 (feasible)**: `H = s_i`; the penalty term vanishes.
- **Weight 0**: `H = A`.
- **Weight k >= 2**: `H = sum of selected scores + A * (k - 1)^2 >= A`.

The QUBO optimum therefore coincides with the constrained optimum exactly when

```text
A > min(s)
```

This condition is genuinely falsifiable. With `s = [0.5, 0.9]` and `A = 0.3`, the all-zeros state has
energy 0.3 against a best feasible energy of 0.5, and `verify_qubo_equivalence` returns
`equivalent = False`. (Small instances report this by the exhaustive method, with the message
*"QUBO optimum does not match the constrained model."*; the analytic path above the enumeration
limit reports *"Penalty is too small to exclude infeasible states."*)

## Verification gate

`verify_qubo_equivalence` runs before any solver output is trusted, by one of two methods:

- **Exhaustive**, for `qubo.size <= exhaustive_limit` (default 20). Enumerates all `2**n` states,
  finds every minimum-energy bitstring, and requires that the minimum energy match `min(s)`, that no
  minimum-energy state be infeasible, and that the decoded optima be a subset of the exact
  co-optimal indices.
- **Analytic**, above that limit. Returns `A > min(s)` directly from the argument above, avoiding an
  intractable enumeration.

`app.py` treats `equivalent = False` as fatal: it renders the verification JSON and returns without
calling a single solver.

## Reference instance

Kenya → Tanzania, `cc1` tier, default policy, inferred `cash_oriented` profile, five model
candidates, `A = 2.0`.

```text
n      = 5
scores = [0.1631558700780258, 0.1763382326256578, 0.2,
          0.25926739248775177, 0.32]
A      = 2.0
offset = 2.0
```

Diagonal `Q[i][i] = s_i - A`:

```text
[-1.8368, -1.8237, -1.8000, -1.7407, -1.6800]
```

All ten off-diagonal entries equal `2A = 4.0`.

### Energy spectrum

Complete enumeration of all 32 states:

| Hamming weight | States | Minimum energy | Maximum energy |
| --- | --- | --- | --- |
| 0 | 1 | 2.0000 | 2.0000 |
| **1 (feasible)** | **5** | **0.1632** | **0.3200** |
| 2 | 10 | 2.3395 | 2.5793 |
| 3 | 10 | 8.5395 | 8.7793 |
| 4 | 5 | 18.7988 | 18.9556 |
| 5 | 1 | 33.1188 | 33.1188 |

Every feasible state lies below 0.32; every infeasible state lies at or above 2.0. The penalty gap
is **1.68**, roughly 10.7x the spread of the feasible band. Verification returns `equivalent = True`
by the exhaustive method.

## Ising conversion

`qubo_to_ising` converts the upper-triangular QUBO to Z and ZZ coefficients under
`x_i = (1 - z_i) / 2`:

```text
h_i    -= Q[i][i] / 2
h_i    -= Q[i][j] / 4        for each j > i
h_j    -= Q[i][j] / 4        for each i < j
J_ij    = Q[i][j] / 4
```

Terms at or below `1e-12` in magnitude are skipped so they never become a gate. The constant offset
is not carried into the Hamiltonian; `QuboModel.energy` adds it back when samples are evaluated.

For the reference instance:

```text
h = [-3.0816, -3.0882, -3.1000, -3.1296, -3.1600]
J = 1.0 for all 10 pairs
```

The uniform `J` follows directly from the uniform `2A` off-diagonal: the one-hot constraint is
symmetric across candidates, so every pairwise coupling is identical. Only the local fields carry
candidate-specific information, and here they span just 2.5% of their own magnitude. **That narrow
spread, not the constraint, is the real difficulty for a variational sampler** — see
[Reference results](qaoa_experiment.md#reference-results), where QAOA reproduces the constraint but
does not discriminate among the feasible states.

## Interpretation limits

- **The QUBO is not the hard part.** A five-variable one-hot QUBO is solved exactly by enumerating
  32 states in microseconds. Its purpose is to give QAOA a correct and verified target.
- **Single-transfer one-of-N remains classically trivial**, O(n) by direct minimum.
- **Qubit cost is now linear in candidates.** `MODEL_CANDIDATE_CAP` equals `DEFAULT_MAX_QUBITS`
  (20), so at most 20 candidates reach the QUBO. Batch mode can enumerate more plans than that;
  candidates are capped before the QUBO is built.
- **Scores must stay non-negative.** The whole Hamming-weight argument depends on it, and the
  constructor enforces it. An objective admitting negative scores invalidates the analytic path and
  needs a new penalty bound.

## Reproducible check

```bash
python - <<'PY'
import itertools
import numpy as np
from qkash.scoring import build_selection_qubo, verify_qubo_equivalence

scores = [0.1631558700780258, 0.1763382326256578, 0.2,
          0.25926739248775177, 0.32]
qubo = build_selection_qubo(scores, penalty=2.0)

print("encoding:", qubo.encoding, "qubits:", qubo.size)
print("diagonal:", np.round(np.diag(qubo.matrix), 4).tolist())
print("off-diagonal:", qubo.matrix[0, 1], "offset:", qubo.offset)
print(verify_qubo_equivalence(qubo)["equivalent"])

by_weight = {}
for bits in itertools.product((0, 1), repeat=qubo.size):
    by_weight.setdefault(sum(bits), []).append(qubo.energy(bits))
for weight, energies in sorted(by_weight.items()):
    print(weight, len(energies), round(min(energies), 4), round(max(energies), 4))

# The penalty condition can fail, and the gate catches it.
bad = build_selection_qubo([0.5, 0.9], penalty=0.3)
print("under-penalised:", verify_qubo_equivalence(bad)["equivalent"])
PY
```

Requires only pandas and numpy. The Ising values additionally require `qkash.quantum.qubo_to_ising`,
which imports Qiskit.
