# RUNBOOK — full end-to-end reproduction

This reproduces every figure and number in the paper from the canonical inputs in
`shared/data/`. No GPU is required: the released perf/quality tables are the inputs;
the MILPs and plots run on CPU.

## 0. Environment

```bash
pip install -r requirements.txt
```
Requires Python ≥ 3.10. Key packages: `pulp` (MILP, ships the CBC solver), `pandas`,
`numpy`, `pyarrow` (parquet), `matplotlib`. Optional (only to *rebuild* the FAISS
indexes, not needed for any paper result): `sentence-transformers`, `faiss-cpu`.

Every script is run **from inside its own `src/` directory** and writes to that
part's `results/figures` and `results/data`. Inputs are found via
`shared/lib/paths.py`.

## 1. Dependency graph (what must run before what)

```
shared/data/*  (given)
   │
   ├── part1: 01 ──→ shared/pareto/perf_capacity_frontier.csv ──→ part1: 07, 08
   │          03 ──→ results/data/baseline_v4.json            ──→ part1: 09
   │          02, 04, 05, 06  (independent given shared/data)
   │
   ├── part2: 00 ──→ results/data/{agg_frontier, quality_cost_merged,
   │                  dispatcher_f1, policy_compare}.parquet, proxy_vs_quality.csv
   │                                                          ──→ part2: 02, 03
   │          01 ──→ shared/pareto/quality_additive_frontier.csv
   │          04, 05, 06  (independent given shared/data; 05 self-builds its table)
   │
   └── part3: 02 ──→ shared/pareto/quality_gated_frontier.csv ──→ part3: 03
              01  (head-to-head, independent)
```

## 2. Exact order

### Part 1 — static allocation (parameter proxy)
```bash
cd part1_static_allocation/src
python3 01_build_capacity_frontier.py      # ~1–3 min (MILP sweep) → shared/pareto/perf_capacity_frontier.csv
python3 02_baseline_vs_greedy.py           # ~1–2 min → "tie on 95%" verdict
python3 03_heterogeneous_synthetic.py      # ~2–4 min (MILP) → baseline_v4.json (SYNTHETIC affinities)
python3 04_uncertainty_sensitivity.py      # ~1 min  → uncertainty_k3.json
python3 05_performance_figures.py          # secs → heatmap, throughput, quant_cost, cost_landscape
python3 06_family_correlation_figures.py   # secs → family_cost, corr_heatmap
python3 07_frontier_figures.py             # secs → frontier_annotated, chain_breakdown (reads shared frontier)
python3 08_corrected_figures.py            # secs → syscap3_frontier, baseline_cmp, uncertainty
python3 09_heterogeneous_figure.py         # secs → baseline_hetero (needs 03 first)
cd ../..
```

### Part 2 — quality-aware
```bash
cd part2_quality_aware/src
python3 00_prepare_intermediates.py        # secs → builds the 5 intermediate tables (RUN FIRST)
python3 01_quality_aware_milp.py           # ~1–3 min (MILP sweep) → shared/pareto/quality_additive_frontier.csv
python3 02_quality_frontier_analysis.py    # secs → q1,q2,q3,q4 (needs 00)
python3 03_dispatcher_and_domain_plots.py  # secs → q5,q6 (needs 00)
python3 04_synthesiser_and_regimes.py      # secs → q7,q8
python3 05_ragas_and_proxy_vs_quality.py   # ~1 min → q10,q11 (self-builds pipeline comparison)
python3 06_quality_frontier_plot.py        # secs → q9
cd ../..
```

### Part 3 — bilinear gated
```bash
cd part3_bilinear_gated/src
python3 01_gated_milp_headtohead.py        # ~3–5 min (McCormick MILP, exact) → bilinear-vs-additive allocation table
python3 02_gated_frontier_and_comparison.py # ~9–11 min (44-point sweep, exact) → shared/pareto/quality_gated_frontier.csv
python3 03_gated_plots.py                  # secs → q14, q15, q16 (needs 02)
python3 04_coupling_sweep.py               # ~2–4 min (McCormick MILP) → coupling_sweep.json, q17
python3 05_batching_robustness.py          # ~2–4 min → sigma_sweep.csv
python3 06_batching_figure.py              # secs → q18 (needs 05)
cd ../..
```

### lean (Qwen-family robustness; concurrent/batched serving)
```bash
cd lean
pip install -r requirements.txt            # highspy (HiGHS), pydantic, pytest
python3 -m catalog.build_catalog           # rebuilds catalog.json from shared/data/perf_table.parquet;
                                           #   sha256 must equal catalog/catalog.json.sha256 (asserted by verify.py)
python3 -m src.experiment experiments/lean_8gb.yaml      # canonical run; auto-uses catalog/quality.parquet
                                                          #   (measured data) -- meta.json declares the source
python3 -m src.ablations experiments/lean_8gb.yaml        # 4 sweeps -> results/ablations/*.csv
python3 -m src.figures --ablations results/ablations --out figures \
    --run results/lean_8gb/<run_id>        # 4 ablation figures + baselines_comparison.pdf
python3 -m pytest tests/ -q                # 65 tests
cd ..
```
The MILP is HiGHS with `random_seed=0`, single-threaded: solutions, not just
objectives, are deterministic. Total: ~1 minute.

### part 4 (dynamic agentic path on the static optimum)
```bash
python3 part4_dynamic_path/routing_eval.py --embedder hashing   # proxy smoke; runs anywhere
python3 part4_dynamic_path/figures_dynamic.py                   # 3 figures from the result JSONs;
                                                                #   each carries the embedder tag and a
                                                                #   red PROXY stamp unless it is bge-m3
```
`results/dynamic_eval.json` ships as the **hashing-proxy** artifact and is
internally verified by `verify.py` (its summary is recomputed from its own
per-query rows). The paper-grade bge-m3 numbers require ONE step on the
full-codebase machine — see `part4_dynamic_path/data/INPUTS.md`:
```bash
python -m part4_dynamic_path.scripts.build_score_cache --embedder bge-m3
```
after which `routing_eval.py` / `ood_eval.py` replay the frozen real-embedder
scores offline. `run_pipeline.py` and `synth_subset_rerun.py` are GPU-side
(GGUF + llama-cpp-python); see `part4_dynamic_path/RUNBOOK_pipeline.md`.

### Final check
```bash
python3 verify.py   # asserts every headline number (parts 1-3, lean, part 4) against the artifacts
```

Total wall-clock on a laptop CPU: roughly **25–40 minutes**, dominated by the four
MILP sweeps (part1/01, part1/03, part2/01, part3/02). Part3/02 is the single longest
step (~10–15 min): it solves all 44 gated MILPs to **exact** optimality
(`timeLimit=60` per stage, hardest stage-1 instance ~48 s) and then canonicalizes each
solution lexicographically (see README §7), which makes the headline gated numbers
(mean gap 0.18, max 0.32, 42/44) *and the reported allocations* identical to the paper
on any machine.
Script 02 writes its CSV incrementally, so progress is visible and survives a stop.

## 3. Notes for reviewers

- **Run MILP scripts in the foreground.** They issue hundreds of CBC solves; a
  background `nohup` can appear to hang. Foreground is reliable.
- **The shared frontiers are both inputs and outputs.** `shared/pareto/` ships with
  precomputed copies so the plotting scripts work immediately; parts 1/2/3 scripts
  01/01/02 *regenerate* them in place. Either way the downstream plots are correct.
- **Determinism.** The MILPs are deterministic. Bootstrap CIs (part2/03's domain
  figure) use a fixed seed. Re-running yields identical numbers.
- **The synthetic-affinity scripts (part1 03/09) are clearly the synthetic
  demonstration** — they are not measured results. See the paper's §6 and this
  repo's part1/README.md.
- **To rebuild the FAISS indexes** (not required for any paper result):
  `cd shared/faiss_code && python3 build_faiss.py --gold-text ../data/calibration_with_gold.yaml --embedder bge-m3 --out ../corpus`
  (needs `sentence-transformers` + a one-time bge-m3 download). With `--embedder hash`
  it runs fully offline to validate the plumbing.

## 4. Mapping results → scripts

| Paper figure(s) | Script |
|---|---|
| heatmap, throughput, quant_cost, cost_landscape | part1/05 |
| family_cost, corr_heatmap | part1/06 |
| frontier_annotated, chain_breakdown | part1/07 |
| syscap3_frontier, baseline_cmp, uncertainty | part1/08 |
| baseline_hetero | part1/09 |
| q1 (quality-latency frontier), q2 (params≠quality) | part2/02 |
| q3 (proxy validation), q4 (per-domain) | part2/02 |
| q5 (dispatcher F1), q6 (per-domain matrix) | part2/03 |
| q7 (three regimes), q8 (synth quality) | part2/04 |
| q10 (synth RAGAS), q11 (proxy vs quality) | part2/05 |
| q9 (quality-aware frontier) | part2/06 |
| q12 (per-domain CIs) | see part2/README (bootstrap script) |
| q14 (recall vs F1), q15 (gated frontier), q16 (gated vs additive) | part3/03 |
| q17 (coupling sweep) | part3/04 |
| q18 (batching robustness) | part3/06 (data from part3/05) |
