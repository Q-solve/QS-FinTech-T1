# Remit-Q showcase guide

## What the interface demonstrates

The interface presents one fixed, reproducible research case: `2025_3Q`, Kenya
to Tanzania, CC2, KES 45,000, with 14 exact-unique provider/service
alternatives. Zanzibar is part of the story framing but is represented by
Tanzania because the workbook contains no separate Zanzibar destination.

The page deliberately does not expose weight sliders that would make the
stored QAOA results look dynamic. The displayed weights—40% fee, 30% FX margin,
and 30% speed—are the exact weights used by every stored solver run.

## Suggested explanation

1. **Start with the decision.** The audited data is filtered before variables
   are created. Each qubit represents one of 14 provider/service alternatives,
   not one raw spreadsheet row.
2. **Explain the score.** Fee, FX margin, and speed are min–max normalized within
   this eligible set, then combined using weights 0.4, 0.3, and 0.3. Total cost
   is displayed for context but is not added as a fourth objective, which would
   double-count fee and FX margin.
3. **Establish the truth set.** Direct argmin, exhaustive binary enumeration,
   and mathematical optimization agree on the exact optimum. For these fixed
   preferences, Western Union is selected.
4. **Introduce the QUBO.** The same score is encoded with an exactly-one penalty.
   The resulting Ising cost Hamiltonian is shared by both QAOA variants.
5. **Contrast the mixers.** Standard X-QAOA starts over all binary strings and
   can move among infeasible Hamming weights. The alternative begins in a
   uniform one-hot W state and uses ring-XY gates that move one excitation
   without creating or destroying it.
6. **Read the evidence.** Use the p=1/p=2 control. Across five matched seeds, XY
   produced 100% feasible probability and zero ideal leakage at both depths.
   Standard X produced mean feasible probabilities of about 23.2% at p=1 and
   1.1% at p=2. XY also used deeper transpiled circuits.
7. **End with the limitation.** This one-of-14 choice is classically trivial and
   the local simulator timings do not show quantum speedup or quantum
   advantage. The contribution is a validated constraint-preserving
   formulation and measured feasibility improvement.

## Application flow

```text
Audited CSV
   → fixed eligible comparison set
   → normalized weighted score
   → exact classical baselines
   → shared QUBO/Ising cost Hamiltonian
   → standard X or W-state + ring-XY QAOA
   → stored multi-seed JSON results
   → read-only FastAPI showcase payload
   → React evidence dashboard
```

The browser never launches an expensive optimization job. The FastAPI endpoint
rebuilds the deterministic classical ranking from the audited CSV and combines
it with the stored, hash-tracked QAOA comparison artifact. This keeps the live
demonstration fast while preserving the distinction between computed evidence
and interface rendering.

## Live demonstration checklist

- Start FastAPI on port 8000 and Vite on port 5173.
- Confirm the header reports `API online`.
- Point out the KES 45,000 benchmark and Tanzania/Zanzibar disclosure.
- Explain the fixed objective weights and exact classical recommendation.
- Toggle between p=1 and p=2 and compare feasibility, leakage, recovery, and
  circuit depth.
- Scroll through the 14 source-derived alternatives.
- Finish on “What this does not prove.”
