# part3 RUNBOOK

Run from inside `part3_bilinear_gated/src/`. Outputs go to `../results/`.
The MILP scripts issue many CBC solves with McCormick auxiliaries — **run them in
the foreground** (a background `nohup` can appear to hang).

```bash
cd part3_bilinear_gated/src

# 1. Head-to-head: bilinear (gated) vs additive allocation, per budget.
python3 01_gated_milp_headtohead.py          # ~2–4 min
#   → console table (allocations differ at every budget),
#     results/data/bilinear_vs_additive.json

# 2. The gated frontier + fair comparison (MILP sweep). Writes the shared frontier.
python3 02_gated_frontier_and_comparison.py  # ~3–6 min → shared/pareto/quality_gated_frontier.csv
#   Console: bilinear ≥ additive (under true gating) at nearly all points.

# 3. Figures (reads step 2's frontier and the quality table).
python3 03_gated_plots.py                    # secs → q14, q15, q16

# 4. Coupling sweep: when does optimization beat greedy? (MILP sweep over lambda)
python3 04_coupling_sweep.py                 # ~4–6 min → q17, coupling_sweep.json
#   The MILP-vs-greedy gap grows ~linearly with coupling, 0.06 -> 0.18.

# 5. Batching robustness: sequential (worst case) vs batched specialists.
python3 05_batching_robustness.py            # ~3–5 min → q18, sigma_sweep.csv
#   Serialization factor sigma in [1/k, 1]; the small-specialist optimum is
#   invariant across the realistic budget range (eps<=20s) in all regimes.
```

## Dependencies within part 3
- **02 → 03** (03 plots the gated frontier 02 writes to `shared/pareto/`).
- `01` is an independent head-to-head (does not need 02).

## Expected headline results (exact, deterministic)
- Bilinear vs additive: **different allocation at every budget tested**.
- Gated optimum's router: **SmolLM2-360M** (global F1 0.39, but ~0.99 recall on the
  activated domains succ_legittima / succ_testamentaria / tutela_minori).
- Specialist stays **Llama-3.2-1B**; synthesiser large (Mistral-7B) at higher budgets.
- Gated frontier: **44 points** across k∈{1,3,5}. Under true gating the gated optimum
  **dominates the additive optimum at 42/44 budgets, mean gap 0.18, max 0.32** — the
  exact numbers in the paper.

## Determinism (why the numbers are exactly reproducible)
The MILP solves use `timeLimit=60`. The hardest single instance (k=5, large ε) takes
~48 s to prove optimality, so 60 s guarantees the **true optimum** on every instance.
This makes the result identical on any machine. (An earlier draft used `timeLimit=8`,
which is *faster* but non-deterministic — under an 8 s cutoff some instances stop
before optimality and *which* ones depends on CPU speed, so the mean gap drifted
between machines. The 60 s setting removes that and yields the paper's numbers exactly.)
Script `02` writes the CSV **incrementally**, so progress is visible and survives
interruption.
