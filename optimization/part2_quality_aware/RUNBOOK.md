# part2 RUNBOOK

Run from inside `part2_quality_aware/src/`. Outputs go to `../results/`.

```bash
cd part2_quality_aware/src

# 0. Build the intermediate tables FIRST (fast). Needed by 02 and 03.
python3 00_prepare_intermediates.py          # secs
#   → results/data/: quality_cost_merged.parquet, agg_frontier.parquet,
#     dispatcher_f1.parquet, policy_compare.parquet, proxy_vs_quality.csv

# 1. The quality-aware MILP sweep. Writes the shared additive frontier.
python3 01_quality_aware_milp.py             # ~1–3 min → shared/pareto/quality_additive_frontier.csv
#   Console shows the recovered optimum: mistral-7b | llama3.2-1b | mistral-7b

# 2–6. Figures.
python3 02_quality_frontier_analysis.py      # secs → q1, q2, q3, q4   (needs 00)
python3 03_dispatcher_and_domain_plots.py    # secs → q5, q6           (needs 00)
python3 04_synthesiser_and_regimes.py        # secs → q7, q8
python3 05_ragas_and_proxy_vs_quality.py     # ~1 min → q10, q11       (self-builds its table)
python3 06_quality_frontier_plot.py          # secs → q9               (reads shared additive frontier)

# 7. Robustness figures.
python3 07_confidence_and_weight_sensitivity.py  # secs → q12, q13
```

## Dependencies within part 2
- **00 → 02, 03** (02/03 read the intermediate parquet/csv tables 00 builds).
- **01 → 06** (06 plots the additive frontier 01 writes to shared/pareto).
- **04, 05, 07** are independent given `shared/data/` (05 builds its own comparison;
  07 reads the quality table directly).

## Expected headline numbers
- Best specialist **Llama-3.2-1B**, mean quality ≈0.74, beating Mistral-7B (≈0.73).
- Quality-aware optimum ≈1.61 vs all-Mistral ≈1.59 vs all-Llama ≈1.45.
- Proxy vs quality-optimal: proxy sub-optimal at every budget, **mean gap ≈0.16**.
- Per-domain CIs: **7/9 domains n<10** (flagged); pooled Llama-1B CI ≈[0.743,0.764].
- Weight sensitivity: small specialist optimal in **all 15** (w_d,w_y) combinations.

## Note on 05
`05_ragas_and_proxy_vs_quality.py` builds the **pipeline-level** proxy-vs-quality
comparison internally (dispatcher + k·specialist + synth at matched latency), which
is distinct from the per-slot `proxy_vs_quality.csv` that `00` writes. Both are
correct for their respective figures.
