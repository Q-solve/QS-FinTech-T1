# Validated QUBO formulation

This document specifies the QUBO that QKash builds from the classical model, the penalty condition
that makes the two equivalent, the Ising conversion used by the QAOA layer, and the verification
that gates every run. It describes `build_selection_qubo`, `verify_qubo_equivalence`, and
`qubo_to_ising` as implemented.

## Scope and shared source model

The QUBO encodes exactly the constrained problem in
[`classical_model.md`](classical_model.md#mathematical-formulation) — same candidate set, same
weighted scores, same one-of-N constraint. It introduces no new objective and no new data. Any
difference in outcome between the classical baselines and QAOA is therefore attributable to the
solver, not to the model.

## Decision variables and constrained objective

Let `s ∈ R^n` be the weighted scores of the `n` modelled candidates, with `x_i = 1` meaning
candidate `i` is selected.

```
minimize    sum_i s_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

The one-hot constraint is folded into the objective with a quadratic penalty `A > 0`:

```
H(x) = sum_i s_i * x_i + A * (sum_i x_i - 1)^2
```

Expanding the squared term gives the upper-triangular QUBO that `build_selection_qubo` returns:

```
Q[i][i] = s_i - A
Q[i][j] = 2A          for i < j
offset  = A
```

`QuboModel.energy` evaluates `offset + sum_{i<=j} Q[i][j] * x_i * x_j`.

`build_selection_qubo` rejects an empty candidate list, rejects any score below `-1e-12`, and
rejects a non-positive penalty. When no penalty is supplied it defaults to `max(1, 2 * max(s))`.

## Penalty condition

Because all scores are non-negative, the energies partition cleanly by Hamming weight:

- **Hamming weight 1 (feasible).** `H = s_i`. The penalty term vanishes.
- **Hamming weight 0.** `H = A`.
- **Hamming weight k ≥ 2.** `H = sum of selected scores + A * (k - 1)^2 ≥ A`.

The QUBO optimum coincides with the constrained optimum whenever

```
A > min(s)
```

This is a weak condition, and `app.py` clears it by a wide margin. It computes
`base_penalty = max(1, max(s))` and multiplies by a user-controlled `penalty_multiplier`
(1.1 to 5.0, default 2.0). With weighted scores in `[0, 1]` the base is pinned at 1.0, so the
default penalty is 2.0 against scores below 0.2.

## Verification

`verify_qubo_equivalence` runs before any solver and uses one of two methods:

- **Exhaustive**, for `n <= exhaustive_limit` (default 20). Enumerates all `2^n` bitstrings, finds
  every minimum-energy state, and requires both that the minimum energy matches `min(s)` and that
  the set of selected indices matches the exact solver's co-optimal set.
- **Analytic**, above that limit. Returns `A > min(s)` directly from the argument above, avoiding an
  intractable enumeration.

At the shipped `TARGET_QUBITS = 5` the exhaustive path always runs — 32 states.

`app.py` treats a false result as fatal: it renders the verification JSON and returns without
calling a single solver. This is the gate that keeps an unverified QUBO from producing a published
number.

## Reference instance

The five-candidate Kenya → Tanzania instance from
[`classical_model.md`](classical_model.md#reference-instance), with `penalty_multiplier = 2.0`.

```
n        = 5
scores   = [0.059218477950031836, 0.06225822738678684, 0.07129763664836268,
            0.09071402607098497,  0.11691012797116852]
A        = 2.0
offset   = 2.0
```

The energy spectrum and verification output below are computed from these full-precision scores.
Rounding them to six places shifts the Hamming-weight-2 minimum by 1e-6.

Diagonal `Q[i][i] = s_i - A`:

```
[-1.940782, -1.937742, -1.928702, -1.909286, -1.883090]
```

All ten off-diagonal entries equal `2A = 4.0`.

### Energy spectrum

Complete enumeration of all 32 states, grouped by Hamming weight:

| Hamming weight | States | Minimum energy | Maximum energy |
| --- | --- | --- | --- |
| 0 | 1 | 2.000000 | 2.000000 |
| **1 (feasible)** | **5** | **0.059218** | **0.116910** |
| 2 | 10 | 2.121477 | 2.207624 |
| 3 | 10 | 8.192774 | 8.278922 |
| 4 | 5 | 18.283488 | 18.341180 |
| 5 | 1 | 32.400398 | 32.400398 |

Every feasible state lies below 0.117; every infeasible state lies at or above 2.0. The gap between
the worst feasible state and the best infeasible state is **1.883**, roughly 16× the spread of the
feasible band itself. The penalty is comfortably sufficient, and a sampler has to make a large error
to leave the feasible subspace.

Verification result:

```
equivalent          True
method              exhaustive
original_objective  0.059218477950031836
qubo_objective      0.05921847795003177
original_indices    [0]
qubo_indices        [0]
```

The `6e-17` difference between the two objectives is float accumulation in `QuboModel.energy`, well
inside the `numpy.isclose` tolerance used for the comparison.

## QUBO-to-Ising conversion

`qubo_to_ising` converts the upper-triangular QUBO to Z and ZZ coefficients under the standard
substitution `x_i = (1 - z_i) / 2`:

```
h_i    -= Q[i][i] / 2
h_i    -= Q[i][j] / 4        for each j > i
h_j    -= Q[i][j] / 4        for each i < j
J_ij    = Q[i][j] / 4
```

Terms with magnitude at or below `1e-12` are skipped so they never become a gate. The constant
offset is not carried into the Hamiltonian; QAOA optimizes against `QuboModel.energy`, which adds it
back, so the offset never affects the reported objective.

For the reference instance:

```
h = [-3.029609, -3.031129, -3.035649, -3.045357, -3.058455]
J = 1.0 for all 10 pairs
```

The uniform `J` is a direct consequence of the uniform `2A` off-diagonal: the one-hot constraint is
symmetric across candidates, so all pairwise couplings are identical. Only the local fields carry
candidate-specific information, and they differ by less than 1% of their magnitude. That narrow
spread is the real difficulty of this instance for a variational sampler — not the constraint, which
is trivially enforced, but the near-degeneracy of the five feasible states.

## Interpretation limits

- **The QUBO is not the hard part.** A five-variable one-hot QUBO is solved exactly by enumerating
  32 states in microseconds. Its purpose is to give QAOA a correct and verified target, not to pose
  a challenge.
- **Penalty tuning is not a research result.** The sufficient condition `A > min(s)` is met by every
  value the UI slider can produce. Reporting penalty sensitivity as a finding would overstate it.
- **Scores must stay non-negative.** `build_selection_qubo` enforces this, and the whole
  Hamming-weight argument above depends on it. A future objective that admits negative scores
  invalidates the analytic verification path and requires a new penalty bound.

## Reproducible command

```bash
python - <<'PY'
import itertools
import numpy as np
from qkash.scoring import build_selection_qubo, verify_qubo_equivalence

# Full-precision scores from the reference instance in classical_model.md.
scores = [
    0.059218477950031836,
    0.06225822738678684,
    0.07129763664836268,
    0.09071402607098497,
    0.11691012797116852,
]
qubo = build_selection_qubo(scores, penalty=2.0)

print("diagonal:", np.round(np.diag(qubo.matrix), 6).tolist())
print("off-diagonal:", qubo.matrix[0, 1])
print(verify_qubo_equivalence(qubo))

by_weight = {}
for bits in itertools.product((0, 1), repeat=qubo.size):
    by_weight.setdefault(sum(bits), []).append(qubo.energy(bits))
for weight, energies in sorted(by_weight.items()):
    print(weight, len(energies), round(min(energies), 6), round(max(energies), 6))
PY
```

Requires only pandas and numpy. The Ising values additionally require `qkash.quantum.qubo_to_ising`,
which imports Qiskit.
