# Part 4 — Dynamic, quality-aware optimization of the agentic path

This module adds a **dynamic agentic-path** layer on top of the static allocation
of Parts 1–3. It keeps the pipeline structure unchanged (dispatcher → specialists
→ synthesiser), keeps the fixed per-role models from the Part-2/3 optimum, adds
**no second LLM**, and uses **no risk label** and **no dispatcher confidence**
(the deployed dispatcher exposes neither). It decides, *per query*, **which
specialist domains actually run** — making the number of activated specialists
`k(q)` endogenous instead of a swept parameter.

## Idea in one line
A training-free, LLM-free selector recasts router-gated specialist pruning as
**resource selection** (which corpora to search) with a **measured-quality**
objective and a **conformal coverage guarantee**, on the measured on-device cost
model from Part 1.

## Signals (all LLM-free, computed at runtime)
- **Retrieval geometry** — similarity of the query to each domain's corpus
  passages (the same FAISS retrieval the specialist would run anyway).
- **Description geometry** — similarity of the query to each agent's text
  description (a cheap, always-available domain prototype).
- Offline only: **measured `Q_spec(M_s, domain)`** (quality table) and per-domain
  reliability from Part 3 (optional tempering).

Per-domain corpus profiles are built once, offline, by attributing each gold
authority in the **calibration** set to the domain(s) that cited it
(`calib_clean_with_gold_text.yaml`) — calibration only, no test leakage.

## Selector
Objective (legal-appropriate): **minimize cost(S) subject to a conformal
coverage floor** on relevant domains. Keep every domain whose calibrated
relevance `rho_hat >= tau` (tau set by conformal risk control so the expected
miss-rate of relevant domains `<= alpha`), then add domains by **cost-aware
submodular coverage** (facility-location greedy, `1 − 1/e`) so relevant-but-
redundant domains are skipped. Baselines for the head-to-head: `full`, `topk`,
`threshold`.

## Files
```
part4_dynamic_path/
  src/
    signals.py            per-domain relevance from retrieval + descriptions
    calibrate.py          isotonic relevance map + conformal coverage threshold
    selector.py           conformal-floor + submodular-coverage greedy selector
    costq.py              measured cost model (section-10.5 concurrent Λ) + Q_spec + budget
    run_dynamic.py        end-to-end selection over the test set + baselines + metrics
    synth_subset_rerun.py THE one new experiment (needs real models): synth on subsets
  data/
    test_queries_en.yaml  the 64-query test set (15 out-of-domain → abstain)
    quality_table.parquet measured quality (from Parts 1–3)
  results/
    dynamic_eval.json     selection metrics (written by run_dynamic.py)
```

## How to run

**Selection evaluation (model-free; runs anywhere).**
```bash
# production: uses bge-m3 if installed + reachable, else falls back to a PROXY
python part4_dynamic_path/src/run_dynamic.py --embedder bge-m3
# force the offline proxy (no model download): dependency-free hashing embedder
python part4_dynamic_path/src/run_dynamic.py --embedder hashing
```
Reports, per selector: routing precision/recall/F1 vs gold, **abstain accuracy**
on the 15 out-of-domain queries, mean `k`, mean latency/energy, and latency
reduction vs full activation. By default candidates = all 9 domains so the gate
itself must decide to abstain; pass `--use_router_candidates` to prune *within*
the dispatcher's set instead.

**Final-answer validation (needs the real models).**
```bash
python part4_dynamic_path/src/synth_subset_rerun.py --spec_model llama3.2-1b --synth_model mistral-7b
```
Re-runs ONLY the synthesiser on subsets (FULL / leave-one-out / dynamic set) of
the already-stored specialist outputs and scores correctness vs gold. This is
the experiment that fits the coverage/value weights and *proves* pruning
preserves the final answer. Adapt the two backend calls (`load_backend`,
`synth.generate`) and the synth prompt to your project's API.

## Decisions locked (by data, earlier in the project)
- **Signal basis:** retrieval geometry (per query) + measured quality / Part-3
  reliability (offline). No confidence, no risk. Retrieval-separation
  precondition checked (AUC ≈ 0.76 under the *proxy* embedder; a gold domain is
  rank-1 84% / top-3 100% of the time) — viable; bge-m3 should do at least this.
- **Objective:** minimize cost subject to conformal coverage (safety-first).
- **Selector:** submodular coverage primary; threshold/topk/full are baselines.

## Honest caveats
- Without bge-m3 the runner uses the **hashing-embedder proxy**, which has no
  semantics — it cannot tell out-of-domain queries are irrelevant, so abstain
  accuracy is low under the proxy. **Run with bge-m3** for the real numbers; the
  abstain mechanism itself is verified to work (it does choose `k=0`).
- The conformal guarantee is **marginal** (coverage on average over queries),
  honest at ~37 calibration queries; achieved coverage is reported on test.
- `synth_subset_rerun.py` is the gate on every *final-answer* quality claim. The
  selection metrics here do not by themselves prove answer preservation.
- Validate the synthesiser produces coherent Italian before trusting its scores
  (early stored synth outputs were degraded boilerplate).
```
