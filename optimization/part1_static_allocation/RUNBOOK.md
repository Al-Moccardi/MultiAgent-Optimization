# part1 RUNBOOK

Run from inside `part1_static_allocation/src/`. Outputs go to `../results/`.
MILP scripts are slow-ish; run them in the foreground.

```bash
cd part1_static_allocation/src

# 1. The capacity-latency frontier (MILP sweep). Writes the shared frontier.
python3 01_build_capacity_frontier.py        # ~1–3 min

# 2. Honest baseline: greedy sort vs MILP.
python3 02_baseline_vs_greedy.py             # ~1–2 min  → "tie on 95%"

# 3. Synthetic heterogeneous demo (MILP). MUST run before 09. (SYNTHETIC affinities.)
python3 03_heterogeneous_synthetic.py        # ~2–4 min  → results/data/baseline_v4.json

# 4. Uncertainty / sensitivity of the frontier.
python3 04_uncertainty_sensitivity.py        # ~1 min    → results/data/uncertainty_k3.json

# 5–6. Performance-characterization figures (fast).
python3 05_performance_figures.py            # secs → heatmap, throughput, quant_cost, cost_landscape
python3 06_family_correlation_figures.py     # secs → family_cost, corr_heatmap

# 7. Frontier figures (reads shared/pareto/perf_capacity_frontier.csv from step 1).
python3 07_frontier_figures.py               # secs → frontier_annotated, chain_breakdown

# 8. The paper's frontier/baseline/uncertainty figures.
python3 08_corrected_figures.py              # secs → syscap3_frontier, baseline_cmp, uncertainty

# 9. Synthetic-heterogeneous figure (reads step 3's json).
python3 09_heterogeneous_figure.py           # secs → baseline_hetero
```

## Dependencies within part 1
- `07`, `08` read the frontier produced by **01**.
- `09` reads `baseline_v4.json` produced by **03**.
- `02`, `04`, `05`, `06` are independent given `shared/data/`.

## Expected headline numbers
- Baseline: MILP strictly beats greedy on ~7/132 points; **tie on ~125/132 (95%)**.
- Synthetic heterogeneous: MILP strictly beats greedy on ~100/132 points (**76%**).
- Frontier is memory-bound near ≈12 B distinct parameters and shifts with k.
