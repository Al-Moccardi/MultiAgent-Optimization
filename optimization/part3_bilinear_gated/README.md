# part3_bilinear_gated — the gated (bilinear) MILP

**Question:** do the roles interact? A specialist's answer is only produced if the
dispatcher routes to it — so its contribution should be **gated** by the router's
recall on that domain.

**Answer:** yes, and it matters. Modelling the coupling as a **bilinear** term
(router × specialist), solved exactly on measured data, changes the optimal
allocation at nearly every budget — and reveals that the right router is the one
with high recall **on the activated domains**, not high global F1.

## The model

For activated domain δ, the specialist contribution becomes the bilinear product

```
ρ[m_d, δ] · Q_spec[m_s, δ]
```

where `ρ[m_d, δ]` is the dispatcher model's **measured per-domain routing recall**
(estimated from the dispatcher logs in `quality_table.parquet`) and `Q_spec[m_s, δ]`
is the specialist model's measured quality on δ. The product of two decision
variables is linearized **exactly** with McCormick envelopes at the model level
(`w ≤ y_d`, `w ≤ y_s`, `w ≥ y_d + y_s − 1`), keeping it a MILP. This is the measured
counterpart of Part 1's *synthetic* heterogeneous demonstration.

## Key findings reproduced here

- **Coupling changes the optimum** (01): the gated MILP picks a different allocation
  from the additive one at every budget tested.
- **A small, high-recall router wins** (q14): the gated model selects
  **SmolLM2-360M** — low global F1 (0.39) but ~0.99 recall on the activated domains —
  because gating values recall on the *used* domains, and the tiny router frees
  budget for the other roles. The specialist stays small (Llama-1B).
- **Gated frontier** (q15): the gated quality–latency Pareto frontier for k∈{1,3,5}.
- **Fair comparison** (q16): the *additive* optimum, re-scored under the **true**
  gating, is dominated by the gated optimum at 42/44 budgets (mean gap **0.18**, max
  **0.32**) — the exact numbers in the paper.

## Scripts (`src/`)

| Script | Output |
|--------|--------|
| `01_gated_milp_headtohead.py` | console table: bilinear vs additive allocation per budget; `results/data/bilinear_vs_additive.json` |
| `02_gated_frontier_and_comparison.py` | `shared/pareto/quality_gated_frontier.csv` (MILP sweep + fair re-scoring) |
| `03_gated_plots.py` | q14, q15, q16 (needs 02) |

## Honesty flags

- `ρ[m, δ]` is estimated from **few queries per domain** → the *specific* winning
  router (SmolLM2-360M) is indicative. The **structural** results are robust:
  (1) coupling changes the optimum, (2) recall-on-activated-domains (not global F1)
  is the right quantity, (3) the gated optimum dominates the additive one under true
  gating.
- McCormick is applied at the **model** level (≈3×11×11 aux vars); a config-level
  encoding would be ~40× larger and is unnecessary (ρ and Q_spec depend on the model,
  not the quantization).

See `RUNBOOK.md`. **02 must run before 03.**
