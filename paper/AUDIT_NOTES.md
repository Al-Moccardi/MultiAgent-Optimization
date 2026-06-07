# AUDIT NOTES — serial draft vs. MAMAP v3 verified package (2026-06-05)

Scope: uploaded `mamap_serial` draft (51 pp) audited line-by-line against the
verified v3 repository. Every change below is traceable to a file+number.

## 1. v1-stale core numbers (§5–§10) — 20-hunk port of the reviewed edit program
Source: `replication/AUTHOR_REVIEW_DIFFS.md` (paper section). Headline fixes:
throughput floor 8→29 tok/s (granite MoE); Q3→Q8 cost 27%→18% slowdown;
heatmap exclusions "nine unavailable / four over" → fifteen excluded / zero over
budget; correlation matrix r(energy,lat)=0.94→0.73, r(params,energy)→0.94,
params–VRAM 0.85→0.84, params–thr −0.80→−0.79, TTFT "≤0.52"→0.80 mechanical /
≤0.45 elsewhere; §7 k=1 "≈12B at 9s"→"11.9B at 9s, 12.2B ceiling at ≈13s";
Table 3 rows → canonical (lexicographic tie-break, `syscap3_frontier.csv`);
§8.1 r=0.57→0.56, Mistral 0.73→0.72; §8.3 "39 budgets/0.049"→"105 steps/0.051";
§8.6 dispatcher 0.68/0.62/0.62→0.65 (Phi) /0.59 (Mistral) /0.57 (Gemma),
llama3.2-3b 0.24→0.056 n=312 degenerate; duplicated groundedness sentence
removed; §9.3 fixed policies 1.61/1.59/1.45→1.60/1.60 (Δ=0.002, disclosed)/1.44;
proxy gap mean 0.16/max 0.25→0.13/0.18 (abstract+§9.3+conclusion);
§10.4 λ=0 gap 0.06→0.05, slope 0.12→0.13; §10.5 σ-sweep 100/100/96/84.
Evidence: `shared/data/{perf_table.parquet,quality_scorecard.csv,quality_table.parquet}`,
parts 1–3 result CSVs, `verify.py` (131 checks).

## 2. §12.4 — synthetic-run contamination (CRITICAL)
Draft text reported Q=1.50, L=6.47s, 6.19GB, alloc 3b/1.5b/0.5b: these are the
synthetic-quality fallback run (experiment launched without the measured-quality
flag), byte-identical to the fallback log. Canonical measured solution
(`lean/results/lean_8gb/20260605T183555Z/alloc.json`): Q=1.88, L=8.00s (SLA T°=8
binds), 5.43GB, Qwen2.5-3B Q8 dispatcher + single shared Qwen2.5-1.5B Q5 group
(2 loaded groups). Baselines (`baselines.csv`): largest-fits 1.39, uniform 1.60,
random-feasible 1.88; per-role-best Q=1.95 INFEASIBLE over SLA only (L=10.97s,
memory 6.37 fits) — draft's "11.3s" corrected to 11.0s. Fig 37 already plotted
the measured CSV (concurrent 1.71@4s, plateau ≈1.95, sequential infeasible
4–6s): the text/figure mismatch is resolved in favour of the figure; figure
replaced with the canonical `lean/figures/sequential_vs_concurrent.pdf`.

## 3. §10.5 exact-concurrent — boundary correction
Re-solved in-container (CBC, grid ε∈{4..28}): concurrent feasible from 4s
(14/14), sequential from 7s (11 pts), plateau Q=1.5991 both, concurrent ≥
sequential everywhere — all confirmed. But specialist = Llama-3.2-1B only at
13/14 and 9/11; deviations are STRICT (forcing Llama-1B is infeasible): at conc
ε=4 and seq ε=7,8 only SmolLM2-360M fits. Text now claims sub-1.3B at 100% with
the exact split. Fig 29 regenerated from this run.

## 4. §11.1 single-family ablation — formulation misstatement
Draft said "identical to §9" with "≈18 feasible". Forensics: 18 = 3 models × 3
quants × 2 contexts ⇒ c_min≥4096 was enforced, and only the CONCURRENT chain +
c_min reproduces the draft exactly: specialist 1.5B for ε∈[5,10.5], flips to 3B
at ε=11.0, plateau Q=1.5112 (= Fig 32's 1.511); tight-budget allocation = 3B
dispatcher + shared 1.5B spec/synth (same structure as §12.4 canonical).
Formulation sentence rewritten accordingly; within-family quality numbers
(0.754/0.736; 0.493→0.679→0.716; 0.659→0.603; 0.754→0.712) verified EXACT under
pooled per-record means and now labelled as such (scorecard means 0.74/0.72 rank
identically). Figs 31–32 regenerated from verified data.

## 5. §11.2 coupling collapse
Re-solved (gated vs additive, Qwen pool, integer ε 4–40): identical
dispatcher+specialist at EVERY feasible budget under BOTH formulations — 33/33
(sequential, from ε=8) and 37/37 (concurrent+c_min, from ε=4). Draft's "27
feasible budgets" was a grid artifact → replaced with measured counts. qwen3b
global F1 0.4287 ("0.43" ✓; old Fig 18's 0.48 was the stale side — vetted q5
figure now agrees with text). Activated-domain recall 0.895 macro / 0.887 micro
→ draft's "0.84" corrected to 0.89. "up to 0.35" → 0.32 (max 0.3239, §10.3).

## 6. Figures
30 shared-core figures replaced with vetted v3 copies (`paper/fig/`);
fig_seq_vs_conc, fig_family_quality, fig_family_sweep regenerated from the
verification runs above; dyn_lean_concurrent ← canonical lean figure;
dyn_routing_main / dyn_beta_sweep / dyn_calib_ablation / dyn_ood kept (author
bge-m3 runs, consistent with retained §12 text).

## 7. Kept with provenance (unverifiable in-package today)
§12 dynamic-routing columns (F1 0.50, abstention 0.60, calibration ablation)
are bge-m3 author-run numbers consistent with part4's README; the package
replays them offline once the one-time score cache is committed (sentence added
to Reproducibility). Static Table 5 latencies (12.18/11.87/10.67) verified
in-repo via the proxy run. The 94-query calibration set is referenced but not
shipped — must be added before submission.

## Verdict
Uploaded draft: 4.5/10 (excellent narrative arc and honesty culture; dozens of
stale numbers, one section reporting a synthetic fallback run as measured,
internal text↔figure contradictions). Post-fix: ≈8.5/10. Residuals: §12 bge-m3
numbers await the score cache; rig calibration remains author testimony.

# MoE REMOVAL PASS (2026-06-05, second pass)

Directive: remove the granite-3.1-3B MoE (`ministral-3b`) from the paper and
every plot, implicit and explicit. Implementation: filtered repo copy
(`perf_table` 84→75 configs, `quality_scorecard` 560→500, `quality_table`
9016→8050 rows); parts 1–3 fully re-run on the filtered pool; all 29 shared-core
figures regenerated and swapped; Figs 29/31 re-derived in-container; lean (§12)
untouched (Qwen-only). part2/02 had a hardcoded ministral annotation (crash) —
patched and re-run.

Changed numbers (old → new, all traceable to nomoe rerun logs/CSVs):
- Pool: 84→75 configs, 11→10 models, six dense families; records
  8925→7959 (disp 3900, spec 3439, synth 620 over 25 configs).
- §5: floor 29→39 tok/s (Mistral-7B); power law −0.7→−0.6; Q3→Q8 1.6×/18%→
  1.5×/20%; corr matrix rebuilt: params–VRAM 0.85, params–thr −0.80,
  params–lat 0.96, energy–lat 0.73→0.99, params–energy 0.95, TTFT |r| 0.51–0.75
  (no longer independent). Family-outlier paragraph deleted; §5 now states the
  inverse: in the all-dense pool size predicts serving cost tightly.
- §7: ceiling 12.2→11.9 B (plateau trio Mistral+SmolLM2-1.7+Qwen-3B, 6.84 GB);
  Granite plateau row removed from Table 3 (other rows byte-identical to
  perf_capacity_frontier.csv); milestones 9/15/21/33 s; greedy ties
  125/132 (95%)→84/91 (92%), strict-win mean 0.52→0.56 B; heterogeneous MILP
  wins 76%→92%, per-k 17/62/64/100→25/94/100/100.
- §8: r 0.56/0.32→0.54/0.31; Pareto 4/56→4/50 (same four configs);
  consensual-separation winner granite→Mistral-7B (LODO losers now both
  Mistral); synth third granite 0.71→Gemma-2 0.72.
- §9: verified UNCHANGED (plateau 1.599; fixed 1.60/1.60 Δ=0.002/1.44; proxy
  gap 0.126/0.181 over 26 budgets).
- §10: 42/44, mean 0.178, max 0.324 UNCHANGED; λ-sweep 0.05→0.18 slope 0.13 →
  0.06→0.17 slope 0.11; σ-sweep 100/100/96/84 UNCHANGED; §10.5 sub-1.3B
  invariance UNCHANGED (13/14, 9/11).
- Scientific consequence (disclosed to author): the architecture-warning
  evidence is gone; the parameter-proxy critique now rests solely on the
  measured-quality results of §8, and §5's metric-coupling story inverts
  (size does predict cost in a dense pool).

# §12 FINALIZATION — verified against committed artifacts (2026-06-06)

Score provenance closed: `score_cache.json` (bge-m3, 173 queries = 94 calib +
64 test + 15 OOD) + `score_cache_vectors.npz` (frozen domain-profile vectors).
The vector sidecar closes the coverage-add live-embedder leak found during
verification (proxy-vector regime gave k=3.67; frozen-bge regime gives k=3.61;
F1/prec/rec/latency invariant). All §12 numbers now come from one regime,
replayable with no GPU and no embedder:

- Table 5 (routing): static 0.33/0.24/0.93/k6.00, lat 12.18/11.87/10.67;
  dynamic 0.50/0.43/0.86/k3.61, lat 11.07/10.76/9.69 (−9.1/−9.3/−9.2%).
  Sources: routing_eval_concurrent.json (+ gated row cross-checked by both
  sweep files at 10.76, −9.3%, k 3.61).
- Gate: per-domain τ, max 0.166 — byte-identical to the original run.
- Sequential model: −26.9/−27.3/−35.3% (routing_eval_seq.json) → "27–35%";
  abstract/intro "9–35%".
- β sweep (routing_beta_sweep.json): recall 0.803→0.864, k 3.12→3.61,
  F1 0.493–0.503 flat, β=3 knee (β=5 identical).
- Recall-floor sweep (routing_recall_sweep.json): F1 ∈ [0.484, 0.503] → the
  "[0.48,0.50]" claim verified exactly; binds only at floor=1.0.
- Calibration ablation: 25-query row 0.37/0.92/0.39/0.62/τ0.467 reproduced
  exactly by an independent mis-configured run; 94-query row updated to
  0.50/0.43/0.86/3.61/0.166.
- OOD (ood_eval.json, live bge-m3): abstention 0.600 (9/15), mean_false_k 1.2;
  abstains on all Italian-other-branch (4/4) and all non-legal (4/4); leaks =
  foreign jurisdiction (3/4, FR abstains) + malformed (0/3) — matches the
  paper's semantic-gating-limitation claim.
- Figures dyn_routing_main / dyn_beta_sweep / dyn_calib_ablation / dyn_ood
  regenerated in-container from the four uploaded JSONs.
- Release patches shipped: routing_eval.py (truthful "bge-m3 (cached)" label),
  build_score_cache.py (--extra_files + vector sidecar export),
  dynamic_lib.py (sidecar load; MAMAP_REBUILD_VECTORS guard).

Residual to ~9+: seed-average or drop the random-feasible baseline tie in
§12.4 (1.878 vs 1.881).

## Reviewer-repo assembly pass (06 Jun, final)
- §8.7 synthesiser count corrected in tex: "695 scored records over 28
  configurations" was the pre-MoE-removal figure; shipped table has 620
  scored over 25 configurations (3 ministral synth configs × ~25 queries
  removed). Model ranking (0.768/0.729/0.717) and sub-2B range (0.579-0.648
  = "0.58-0.65") verified unchanged. Recompiled: 51 pp, 0 errors.
- Four standalone verification scripts added under optimization/scripts_verify;
  all claims of §8, §10.5, §11.1, §11.2 reproduce exactly (see VERIFY.md).

## Final commit pass (06 Jun, evening)
- §12.4: random_feasible baseline row DROPPED (author decision; it tied the
  canonical allocation within seed noise). "four naive baselines" ->
  "the naive baselines (largest-fits, uniform)". Recompiled: 51 pp, 0 errors.
- Committed frozen §12 state: score_cache.json (173 = 94+64+15),
  score_cache_vectors.npz, data/manifests/ood_queries_en.yaml (15 queries).
- requirements.txt (root, verification) + system/requirements.txt added;
  README rewritten as full project overview.
- Remaining author-side artifact: routing_eval_concurrent.json only (the
  06 Jun routing_eval.json upload was byte-identical to the committed
  sequential run).
