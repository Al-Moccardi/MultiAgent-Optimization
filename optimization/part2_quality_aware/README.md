# part2_quality_aware — the measured-quality MILP

**Question:** is the parameter proxy from Part 1 actually right? Replace it with
**measured** pipeline quality and re-optimize.

**Answer:** no. Quality is non-monotone in model size, the three roles follow three
different scaling laws, and the quality-aware optimum is a *mixed* allocation that
beats the proxy at every latency budget.

## Key findings reproduced here

- **Quality ≠ parameters** (q2): the best specialist is **Llama-3.2-1B**, beating
  Mistral-7B at 6× fewer params; the relation is non-monotone (r≈0.57).
- **Quality–latency frontier** (q1): only ~4 of 56 specialist configs are
  non-dominated, all ≤1.2 B.
- **Validation** (q3): the parameter-maximizing policy is sub-optimal at every
  latency budget.
- **Three scaling laws** (q7): dispatcher (routing F1) and synthesiser want **large**
  models; specialists want **small**.
- **Synthesiser RAGAS decomposition** (q10): correctness ↑ with size, faithfulness ↓
  (large models lean on parametric knowledge); context precision/recall are
  retriever-fixed constants under oracle retrieval.
- **Quality-aware MILP** (q9, q11): objective Q = w_d·F1 + mean specialist quality +
  w_y·synth quality; recovers the mixed optimum (Mistral dispatcher | Llama-1B
  specialist | Mistral synth) and dominates the parameter proxy (mean gap 0.16).
- **Robustness**: per-domain bootstrap CIs with n<10 flags (q12); weight-sensitivity
  showing the small-specialist optimum holds across the (w_d, w_y) grid (q13).

## Scripts (`src/`)

| Script | Output |
|--------|--------|
| `00_prepare_intermediates.py` | builds 5 intermediate tables in `results/data/` — **run first** |
| `01_quality_aware_milp.py` | `shared/pareto/quality_additive_frontier.csv` (MILP sweep) |
| `02_quality_frontier_analysis.py` | q1, q2, q3, q4 |
| `03_dispatcher_and_domain_plots.py` | q5, q6 |
| `04_synthesiser_and_regimes.py` | q7, q8 |
| `05_ragas_and_proxy_vs_quality.py` | q10, q11 (self-builds the pipeline comparison) |
| `06_quality_frontier_plot.py` | q9 |
| `07_confidence_and_weight_sensitivity.py` | q12, q13 |

## Quality metric (recap)

Judge-free, embedding-based: correctness = cos(answer, gold); faithfulness = max
cos(answer, context); answer relevancy = cos(answer, question); aggregate
Q = 0.5·correctness + 0.5·mean(faithfulness, relevancy). Context precision/recall
are diagnostics, excluded from Q. The synthesiser's Q is correctness. Scoring is
under **oracle retrieval** (gold passages as context), isolating generation from
retrieval.

## Honesty flags

- 37 root queries; **7 of 9 domains have n<10** → per-domain winners are indicative
  (CIs in q12), pooled/cross-role claims are robust.
- **No held-out split** → the frontier is an in-sample optimum.
- Objective weights (w_d=0.15, w_y=1.0) are a modelling choice; q13 shows the
  qualitative optimum is weight-robust.

See `RUNBOOK.md`. **00 must run before 02/03; 01 writes the shared additive frontier.**
