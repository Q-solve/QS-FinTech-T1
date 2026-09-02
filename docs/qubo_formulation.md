# Validated QUBO formulation

This document specifies the QUBO that QKash currently builds, how it is checked against the
classical constrained model, and the limits of the compact encoding.

## Source model

After source/destination/amount filtering, optional hard policy constraints, Pareto pruning, ML
policy inference, and optional batch-plan construction, QKash has `n` model candidate rows with
weighted scores `s_i`.

The original constrained model is still one-of-N selection:

```text
minimize    sum_i s_i * x_i
subject to  sum_i x_i = 1
            x_i in {0, 1}
```

For a single transfer each row is a provider/payment/receiving-method service. In batch research
mode each row is a feasible assignment plan for several transfers.

## Compact binary-index variables

The QAOA circuit uses compact binary-index encoding rather than one qubit per row. For `n`
candidates:

```text
q = ceil(log2(n))
state_count = 2^q
```

Each measured bitstring is decoded as a little-endian integer. Integers `0..n-1` map to real
candidate rows; integers `n..2^q-1` are unused basis states and are infeasible. Example: 9 rows use
4 qubits, giving 16 basis states, of which 7 are invalid.

The helper functions are:

- `required_qubits(n)` for `ceil(log2(n))`.
- `index_to_bits(index, q)` for deterministic solver samples.
- `decode_candidate_index(bits, n)` for QAOA and validation output.

## Constructed QUBO

An arbitrary table of `n` independent candidate scores cannot generally be represented exactly as a
quadratic function of only `ceil(log2(n))` binary variables. The implemented compact QUBO is
therefore an optimum-preserving construction, not a full score-table encoding.

`build_selection_qubo` first solves the original constrained model exactly to identify the best
candidate index. It then builds a Hamming-distance well around that binary index:

```text
energy(bits) = best_objective + penalty * HammingDistance(bits, best_bits)
```

The returned `QuboModel` stores this as diagonal QUBO coefficients plus an offset. The unique lowest
energy state decodes to the exact best candidate. Invalid unused states can appear in samples, but
they cannot be the constructed optimum unless the implementation is broken.

This design is useful for running a small binary-index QAOA circuit and verifying end-to-end sample
decoding. It should not be presented as evidence that QAOA discovered the best service without help:
the compact Hamiltonian is built from the exact optimum.

## Verification gate

`verify_qubo_equivalence` runs before any solver output is trusted.

For small circuits it enumerates every basis state, finds the minimum-energy bitstrings, decodes
them, and checks that:

- the QUBO objective equals the exact constrained objective;
- every minimum-energy bitstring decodes to a real candidate;
- every decoded QUBO optimum is one of the exact co-optimal candidate indices.

For larger circuits it uses the stored optimal index from the construction and reports analytic
equivalence. `app.py` treats `equivalent = False` as fatal and displays `QUBO Validation failed`
without running QAOA.

## Ising conversion

`qubo_to_ising` converts the upper-triangular QUBO matrix to Z and ZZ coefficients with the standard
substitution `x_i = (1 - z_i) / 2`:

```text
h_i    -= Q[i][i] / 2
h_i    -= Q[i][j] / 4        for each j > i
h_j    -= Q[i][j] / 4        for each i < j
J_ij    = Q[i][j] / 4
```

Terms with magnitude at or below `1e-12` are skipped. The constant offset is not a gate, but
`QuboModel.energy(bits)` adds it back when QAOA measurement samples are evaluated.

## Interpretation limits

- Single-transfer one-of-N selection remains classically trivial. Exact selection is O(n).
- Batch mode adds combinatorial plan construction and provider concentration limits, but the current
  UI keeps instances intentionally small for local Aer and qBraid simulator runs.
- The compact QUBO reduces qubit count, but it changes the Hamiltonian from a full one-hot score
  landscape to an optimum-preserving binary well. Use the research dashboard as an implementation
  comparison, not a quantum-advantage claim.
- Scores must stay finite and non-negative. `build_selection_qubo` rejects empty candidate lists and
  negative weighted scores.

## Reproducible check

```bash
python - <<'PY'
from qkash.scoring import build_selection_qubo, required_qubits, verify_qubo_equivalence

scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
qubo = build_selection_qubo(scores)

print("qubits:", required_qubits(len(scores)))
print("state_count:", qubo.state_count)
print("invalid_states:", qubo.invalid_state_count)
print(verify_qubo_equivalence(qubo))
PY
```
