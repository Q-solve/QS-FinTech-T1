# Validated QUBO formulation

## Scope and shared source model

This formulation translates the existing classical provider/service-selection
model into a QUBO without changing its data, eligibility, normalization,
weights, speed encoding, or deterministic classical tie rules. The source of
the coefficients is `rank_alternatives`; no provider or winning answer is
hard-coded. See `docs/classical_model.md` for the upstream definitions.

The reference instance is `PERIOD = 2025_3Q`, `CORRIDOR = KENTZA`, benchmark
`CC2` (KES 45,000), and weights `(fee, FX, speed) = (0.4, 0.3, 0.3)`. It has 14
eligible provider/service alternatives. KES 100,000 is not quoted by this
benchmark and is not inferred here.

## Decision variables and constrained objective

Alternatives are ordered lexicographically by their stable alternative ID.
This produces a deterministic mapping that does not depend on CSV or caller
order. For alternative `i`,

\[
x_i = \begin{cases}
1 & \text{if alternative } i \text{ is selected},\\
0 & \text{otherwise.}
\end{cases}
\]

The classical weighted score `c_i` is reused exactly. Qiskit's initial
`QuadraticProgram` is

\[
\min_x \sum_{i=0}^{13} c_i x_i
\quad\text{subject to}\quad
\sum_{i=0}^{13} x_i = 1,
\quad x_i \in \{0,1\}.
\]

The exactly-one equality is named `select_exactly_one`. The QUBO conversion is
performed with an explicit configurable penalty `P`, never Qiskit's automatic
penalty choice:

\[
E(x) = \sum_i c_i x_i + P\left(\sum_i x_i - 1\right)^2.
\]

Using `x_i^2 = x_i` for binary variables gives the inspectable form

\[
E(x) = P + \sum_i (c_i-P)x_i +
       \sum_{i<j} 2P x_i x_j.
\]

Therefore the exported conventional coefficients are:

- constant offset: `P`;
- linear coefficient for `x_i`: `c_i - P`;
- quadratic coefficient for every pair `x_i, x_j`, `i < j`: `2P`.

The exported upper-triangular matrix places the linear coefficients on the
diagonal and the pair coefficients above it. Its energy convention is the
displayed expanded equation, not a symmetric `xᵀQx` convention that would
count off-diagonal terms twice.

Qiskit's internal polynomial retains diagonal quadratic terms before binary
reduction: its linear term is `c_i - 2P`, each diagonal quadratic term is `P`,
and each off-diagonal pair is `2P`. These two representations agree for every
binary state. The test suite checks all `2^14 = 16,384` states against the
independent coefficient evaluator with an absolute tolerance of `1e-10`.

## Reference variable mapping

The mapping below is stable for the audited 14-alternative instance. Bit
strings printed by `inspect_qubo.py` use this left-to-right `x_0` through
`x_13` order.

| Variable | Alternative ID | Provider | Payment → receiving method |
|---|---|---|---|
| `x_0` | `alt-16a0d6dbadda` | United Bank for Africa (UBA) | Bank account transfer → Bank account |
| `x_1` | `alt-20844375cc3e` | M-Pesa | Mobile money → Cash |
| `x_2` | `alt-29fa0176acdd` | ABC Bank | Bank account transfer → Bank account |
| `x_3` | `alt-503fc0f80db8` | KCB Bank | Bank account transfer → Bank account |
| `x_4` | `alt-5e860c308bdb` | M-Pesa | Mobile money → Mobile wallet |
| `x_5` | `alt-60f5f9067629` | EcoBank Rapid Transfer | Bank account transfer → Cash |
| `x_6` | `alt-9243c33ae323` | UBA Africash | Cash → Cash |
| `x_7` | `alt-92a686bc27e8` | EcoBank Rapid Transfer | Cash → Cash |
| `x_8` | `alt-9b44600ea942` | GT Bank | Bank account transfer → Bank account |
| `x_9` | `alt-a95dd80e8eea` | Ecobank | Cash → Cash |
| `x_10` | `alt-ac5be959dbde` | MoneyGram | Cash → Cash |
| `x_11` | `alt-c77e722fb62f` | Development Bank of Kenya | Bank account transfer → Bank account |
| `x_12` | `alt-cc1e0457fb2b` | Western Union | Cash → Cash |
| `x_13` | `alt-f5f6dbf518bd` | MoneyGram | Cash → Mobile wallet |

Qiskit's solution vector `result.x` follows the quadratic-program variable
order `x_0, ..., x_13`. Qiskit computational-basis state labels use the
opposite display convention, `x_13 ... x_0`. Thus the reference solution is
`00000000000010` in documented `x_0 → x_13` order and
`01000000000000` as a Qiskit state label. The code reports both and tests the
reversal explicitly.

When independent enumeration finds several feasible states at the same
minimum energy, it uses the existing classical ranking to choose among them;
it does not introduce a new QUBO tie rule. It deliberately does not prefer a
feasible state over an equal-energy infeasible state, because that would hide
an unsafe penalty through silent repair. The reference winning score is
unique, so neither case affects its result.

## Penalty condition and sensitivity

Let `c_min = min_i(c_i)`. All normalized objectives and all weights are
nonnegative, so every `c_i` is in `[0, 1]`. A feasible one-hot state has energy
`c_i`, and the best feasible energy is `c_min`.

The all-zero state has energy `P`. A state selecting `k ≥ 2` alternatives has

\[
E(x) = \sum_{i:x_i=1} c_i + P(k-1)^2 \ge P.
\]

Consequently, the strict condition

\[
P > c_{\min}
\]

is sufficient to make the all-zero state and every multiple-selection state
more expensive than the best feasible state. Equality is not sufficient
because it can create an infeasible tie. For the reference instance,
`c_min = 0.001257839033`; the explicit default `P = 2.0` is therefore safely
above the actual threshold and also above the full possible score range.

The sensitivity tests deliberately use `P < c_min`, where the all-zero state
wins, and `P > c_min`, where the exact QUBO optimum is feasible. Nonpositive,
non-finite, Boolean, and otherwise invalid penalties produce a clear error.

## QUBO-to-Ising conversion and exact solvers

`QuadraticProgram.to_ising()` converts the unconstrained QUBO to a 14-qubit
Ising operator plus its scalar offset. The exact path uses Qiskit's
`NumPyMinimumEigensolver` through `MinimumEigenOptimizer`. It is a local,
deterministic correctness oracle; it is not QAOA, hardware execution, or a
quantum runtime result.

For the reference instance, four exact methods agree:

1. direct classical weighted argmin;
2. exhaustive exactly-one classical enumeration;
3. independent enumeration of every unconstrained QUBO state;
4. Qiskit's exact NumPy minimum eigensolver.

They select `alt-cc1e0457fb2b`, with QUBO energy and original feasible weighted
score approximately `0.001257839033`. The selected vector is feasible and has
exactly one bit set.

## Reproducible inspection

From the repository root:

```bash
.venv/bin/python -m backend.scripts.inspect_qubo
```

The command prints the mapping, penalty threshold, coefficient counts, Ising
metadata, all-state energy validation, both bit-order conventions, selected
service, QUBO energy, original score, feasibility, and four-method agreement.
Use `--benchmark CC1` or different valid weights to inspect other instances of
the same shared model.

## Interpretation limit

This is a one-of-14 selection problem. Once scores are computed, direct
classical argmin is linear in 14 alternatives and is the meaningful baseline.
The Ising representation uses 14 qubits but does not make the problem hard and
does not demonstrate quantum advantage. No QAOA, simulator performance claim,
hardware result, optimization API, frontend integration, or qBraid execution
is part of this implementation.
