# VERIFY.md — claim → command → artifact

Every quantitative claim in `paper/mamap_full.tex` traces to a command in this
repository and a stored artifact. Commands below are run from the component
root they name. `paper/AUDIT_NOTES.md` is the running audit ledger.

## A. Optimization study (paper §5–§11) — `optimization/`

Data ground truth: `shared/data/perf_table.parquet` (75 configurations,
10 dense models), `shared/data/quality_table.parquet` (8 050 rows, 7 959
scored), `shared/data/quality_scorecard.csv`, `shared/data/catalog_zoo.yaml`.

| Paper claim (key numbers) | Regenerate with | Artifact |
|---|---|---|
| §5 performance landscape: throughput floor ≈39 tok/s, size exponent ≈−0.6, quantization ≈1.5× mem / ≈20 % thr, correlation matrix | `part1_static_allocation/src/05_performance_figures.py`, `06_…`, `07_…` | `fig/heatmap.pdf`, `throughput.pdf`, `quant_cost.pdf`, `corr_heatmap.pdf`, `family_cost.pdf`, `cost_landscape.pdf` |
| §7 capacity MILP: ceiling 11.9 B; capacity milestones at ε = 9/15/21/33 s; Table 3 | `part1_static_allocation/src/01_build_capacity_frontier.py` | `part1_static_allocation/data/syscap3_frontier.csv` (Table 3 rows are byte-identical) |
| §7 baselines: MILP > greedy in 84/91 points, mean gap 0.56 B; heterogeneous 84/91 with per-k win rates 25/94/100/100 % | `src/02_baseline_vs_greedy.py`, `src/03_heterogeneous_synthetic.py` | `data/baseline_cmp.json`, `data/baseline_v4.json` |
| §7 robustness: optimum stable to ±0.5 B under 20 % latency noise | `src/04_uncertainty_sensitivity.py` | `data/uncertainty_k3.json` |
| §8 statistics: counts 7959/3900/3439/620; r = 0.54 (config) / 0.31 (record); Llama-1B 0.74 vs Mistral-7B 0.72, pooled CI [0.743, 0.764] (n = 282); Pareto 4/50 (≤1.2 B); winners 7/9 + 2 Mistral domains; synth 0.77/0.73/0.72, sub-2B 0.58–0.65; LODO 9/9, gap 0.001/0.008, beats Mistral 7/9 (+0.018) | `scripts_verify/verify_sec8_stats.py` | console (each line prints measured vs expected) |
| §8 figures q1–q18 | `part2_quality_aware/src/02…05`, `part3_bilinear_gated/src/03` | `paper/fig/q*.pdf` |
| §9 quality-aware MILP: three regimes, plateau Q = 1.599; fixed-allocation 1.597 / 1.437; capacity-proxy gap mean 0.126, max 0.181 over 26 budgets | `part2_quality_aware/src/01_quality_aware_milp.py` | `part2_quality_aware/data/quality_frontier.csv` |
| §10 gated bilinear: optima differ from additive in 42/44 budgets; gated-quality dominance mean 0.178, max 0.324; coupling sweep λ: gap 0.058→0.166 (slope ≈0.11); σ-robustness: small specialist at 100/100/96/84 % | `part3_bilinear_gated/src/01_gated_milp_headtohead.py`, `02_gated_frontier_and_comparison.py`, `04_coupling_sweep.py`, `05_batching_robustness.py` | `part3_bilinear_gated/data/bilinear_vs_additive.json`, `gated_frontier.csv`, `coupling_sweep.json`, `sigma_sweep.csv` |
| §10.5 sequential vs exact concurrent: conc feasible 14/14 from 4 s, seq 11/14 from 7 s; Llama-1B 13/14 & 9/11 (SmolLM2-360M at conc ε=4, seq ε∈{7,8}); both plateaus 1.5991; conc ⪰ seq everywhere | `scripts_verify/verify_sec10_concurrent.py` | console + `scripts_verify/out/fig_seq_vs_conc.pdf` |
| §11.1 single-family: Qwen specialist = 1.5 B for ε<11 s, switches to 3 B at ε=11.0; plateau 1.5112; tight regime puts both roles on the 1.5 B model under a 3 B dispatcher, with a single shared 1.5 B load at ε = 7.0–8.5 s; family quality table (qwen .493/.679/.716, smollm .659/.603, llama .754/.712) | `scripts_verify/verify_sec11_family.py` | console + `out/fig_family_sweep.pdf`, `out/fig_family_quality.pdf` |
| §11.2 gated collapse: Qwen-3B global F1 0.4287 (≈0.43) AND best ACT recall 0.895/0.887 (≈0.89); SmolLM2-360M F1 0.3886 (≈0.39) with ACT recall 0.989; gated ≡ additive at 33/33 (seq, ε=8–40) and 37/37 (conc+cmin, ε=4–40), gap 0.0000 | `scripts_verify/verify_sec11_gated_collapse.py` | console |

Canonical definitions used throughout (and asserted in the scripts):
latencies `lat_disp = ttft + 15/thr`, `lat_gen = ttft + 384/thr`; memory budget
6.99 GB; one context per (model, quant) group; objective weights W_D = 0.15,
W_Y = 1.0; dispatcher recall parsed from `quality_table` dispatcher rows with
non-empty output AND gold, `'|'`-split minus {A_dispatcher, A_synth};
configuration tie-breaks canonicalized by `shared/lib/lexsolve.lex_refine`.

### Expected console output of the four verify scripts

All four print one line per claim in the form `measured (expect X)`; a clean
run shows every measured value equal to its expectation. Reference outputs:

```
verify_sec8_stats.py      C1 7959/3900/3439/620 · C2 0.54/0.31 · C3 0.74 vs 0.72,
                          CI [0.743,0.764] · C4 4 of 50 (llama1b Q5 c4096/c8192,
                          smollm2-360m Q3 c8192, Q8 c4096) · C5 7/9 (+2 Mistral)
                          · C6 0.768/0.729/0.717 · C7 9/9, 0.001/0.008, 7/9 +0.018
verify_sec10_concurrent   C1 14/14 from 4 s, 11/14 from 7 s · C2 13/14 & 9/11
                          · C3 1.5991/1.5991 · C4 True
verify_sec11_family       C1 family table · C2 switch at 11.0 · C3 pattern 6.5–8.5, shared load 7.0–8.5
                          · C4 1.5112
verify_sec11_gated_collapse  C1 0.4287, 0.895/0.887; 0.3886, 0.989
                          · C2 33/33 and 37/37, gap 0.0000
```

## B. Deployed system & dynamic layer (paper §12) — `system/`

The §12 result artifacts as verified are committed under
`system/part4_dynamic_path/results/data/`:

| Paper claim | Artifact |
|---|---|
| Table 5 dynamic row (concurrent): F1 0.50 / prec 0.43 / rec 0.86 / k = 3.61; latency 11.07/10.76/9.69 s = −9.1/−9.3/−9.2 % vs static; sequential deltas −26.9/−27.3/−35.3 % (paper: "27–35 %") | `routing_eval_seq.json` (sequential run, committed); `routing_eval_concurrent.json` is the one remaining author-side artifact (see AUTHOR_TODO) |
| β sweep: recall 0.803→0.864, k 3.12→3.61, F1 flat, knee at β=3 (β=5 identical), τmax 0.306→0.166, latency −9.3 % constant | `routing_beta_sweep.json` |
| recall-floor sweep: F1 ∈ [0.484, 0.503], floor binds only at 1.0 | `routing_recall_sweep.json` |
| OOD: abstention 0.600 (9/15), mean false-positive k = 1.2; Italian other-branch 4/4 and non-legal 4/4 abstain; leaks BR/DE/ES | `ood_eval.json` |
| gate threshold τmax = 0.166 at β=3 | both sweep files |

### §12 replay (author hardware: Windows + CUDA, BGE-M3)

```powershell
$env:HF_HUB_OFFLINE="1"; $env:PYTHONIOENCODING="utf-8"
python -m part4_dynamic_path.scripts.build_score_cache --embedder bge-m3 `
  --calib_file part4_dynamic_path\data\manifests\calibration_queries_en.yaml `
  --extra_files part4_dynamic_path\data\manifests\ood_queries_en.yaml
python -m part4_dynamic_path.routing_eval                       # concurrent
python -m part4_dynamic_path.routing_eval --sequential
python -m part4_dynamic_path.routing_eval --sweep_beta 1,2,3,5
python -m part4_dynamic_path.routing_eval --sweep_recall_floor 0.7,0.8,0.9,1.0
python -m part4_dynamic_path.ood_eval
```

Expected markers: `[cache] loaded frozen domain-profile vectors`, 173 cached
scores (94 calibration + 64 test + 15 OOD), routed k = 3.61, gate τmax = 0.166.
The cache freezes BOTH scores and domain-profile vectors
(`score_cache_vectors.npz`); without the vector sidecar the coverage-add stage
re-embeds live and k drifts (3.67 proxy / 3.61 bge) while F1/latency are
invariant — this is why the sidecar is part of the committed state.

The frozen state is committed: `part4_dynamic_path/data/score_cache.json`
(173 entries: 94 calibration + 64 test + 15 OOD),
`part4_dynamic_path/data/score_cache_vectors.npz` (domain-profile sidecar) and
`part4_dynamic_path/data/manifests/ood_queries_en.yaml` (15 OOD queries), so
the replay above runs without re-embedding.

§12.4 (lean single-GPU pipeline): canonical allocation and ablation CSVs under
`system/lean/`; Fig. dyn_lean_concurrent regenerates from them. The
`random_feasible` baseline row was dropped from the paper (tied the canonical
allocation within seed noise; author decision 06 Jun).

## C. MoE retention (deliberate)

`ministral-3b` (granite-3.1-3b-a800m, the pool's only MoE) is excluded from
the paper's pool. Removed here from: both catalog YAMLs, `tools/slm_zoo.yaml`,
both `paths.py` parameter dicts, `optimization/shared/data/*` (75 configs /
8 050 rows) and `system/shared/pareto/perf_table.parquet` (84→75). Retained:
(1) inert `ministral-3b` keys in the PAR lookup dicts of some experiment
scripts — reruns proven unaffected; (2)
`system/part4_dynamic_path/data/quality_table.parquet` ships byte-identical to
what the verified §12 runs consumed, because
`specialist_quality_by_domain` falls back to a pooled mean over ALL configs
for sparse domains, so filtering is not provably inert. The optimization-side
tables that ground §5–§11 contain no MoE rows.

## D. Residual non-determinism (stated)

`lex_refine` pins the configuration within the chosen model triple (min
latency, then min memory). A hypothetical exact float tie at MODEL level would
still follow solver order; no such tie occurs in this data
(`shared/lib/lexsolve.py` docstring).
