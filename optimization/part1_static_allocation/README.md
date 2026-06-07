# part1_static_allocation — the parameter-proxy MILP

**Question:** given the measured costs, which model can each role afford, if we
maximize a transparent **capability proxy = distinct-model parameter count** under
the memory and latency budget?

This is the paper's first step. It builds the allocation MILP, traces the
capacity–latency Pareto frontier, and — honestly — shows that for this *sortable*
proxy a greedy sort ties the MILP on 95% of points. Optimization only becomes
essential once the objective is non-sortable, which we demonstrate on **synthetic**
heterogeneous affinities (the measured version is Part 2).

## What the MILP does

- **Decision:** one configuration per role (dispatcher / shared specialist /
  synthesiser) from the 84 feasible configs.
- **Objective:** maximize the sum of distinct-model parameter counts (the proxy).
- **Constraints:** load-once VRAM ≤ 6.99 GB (a model loaded once is counted once);
  sequential latency ≤ ε (dispatcher + k·specialist + synthesiser), parametric in
  the number of activated specialists k.

## Scripts (`src/`)

| Script | Output | Notes |
|--------|--------|-------|
| `01_build_capacity_frontier.py` | `shared/pareto/perf_capacity_frontier.csv` | the MILP sweep over ε and k. Slow (MILP). |
| `02_baseline_vs_greedy.py` | console verdict | greedy sort vs MILP: tie on ~95% of points (the honest baseline). |
| `03_heterogeneous_synthetic.py` | `results/data/baseline_v4.json` | MILP with **SYNTHETIC** per-domain affinities → MILP wins 76%. *Not a measured result.* |
| `04_uncertainty_sensitivity.py` | `results/data/uncertainty_k3.json` | Monte-Carlo perturbation of the cost table; frontier is stable. |
| `05_performance_figures.py` | heatmap, throughput, quant_cost, cost_landscape | performance characterization. |
| `06_family_correlation_figures.py` | family_cost, corr_heatmap | per-family cost + correlation heatmap. |
| `07_frontier_figures.py` | frontier_annotated, chain_breakdown | reads the shared frontier from 01. |
| `08_corrected_figures.py` | syscap3_frontier, baseline_cmp, uncertainty | the frontier/baseline/uncertainty figures used in the paper. |
| `09_heterogeneous_figure.py` | baseline_hetero | needs `03` first (reads its json). |

## Honesty flags (important)

- The objective is a **parameter proxy**, not quality. Part 2 is the measured
  rebuttal showing the proxy is sub-optimal.
- **Scripts 03 and 09 use synthetic affinities**, labelled as such in the paper.
  They show *when* optimization matters, not a measured outcome.

## Outputs land in `results/figures/` and `results/data/`.
See `RUNBOOK.md` for the exact order (01 and 03 must precede 07/08 and 09).
