# MAMAP-Edge — Replication Package

**Memory-, Latency-, and Energy-Aware Allocation of Quantized Small Language Models
to a Multi-Agent Legal-Assistant Pipeline on a Single 8 GB GPU.**

This repository reproduces every result, table, and figure in the paper. It is
organized so a reviewer can run each part independently, from the canonical input
data, with no absolute paths and no hidden state.

---

## 1. The paper in one paragraph

A consumer device has one ~8 GB GPU. We must run an 11-agent Italian family-law
assistant on it: one **dispatcher** (routes a query to specialist domains), nine
**domain specialists** (retrieval-augmented answerers), and one **synthesiser**
(composes the final answer). Each agent can be served by any of 11 open-weight
SLMs (0.36–7.2 B) in various quantizations and context lengths — **84 feasible
configurations**. Which model should serve each role, under the memory and latency
budget? The paper answers this in three escalating steps, which are the three
parts of this repository.

## 2. The three-part arc

| Part | Question | Objective | Key result |
|------|----------|-----------|------------|
| **part1_static_allocation** | What can we afford? | maximize a **parameter-count proxy** for capability, under memory + latency | A clean MILP + Pareto frontier; but a greedy sort ties it on 95% of points (reported honestly). Optimization only becomes essential once the objective is non-sortable (shown on *synthetic* heterogeneous affinities, 76% win). |
| **part2_quality_aware** | Is the proxy right? | maximize **measured** pipeline quality (judge-free RAGAS) | **No.** Quality is non-monotone in size: the best specialist is Llama-3.2-1B, beating Mistral-7B at 6× fewer params. Three scaling laws: dispatcher & synthesiser want **large** models, specialists want **small**. The quality-aware MILP recovers a *mixed* optimum and beats the parameter proxy at every budget (mean gap 0.13).

Two further packages sit alongside parts 1-3 (added 2026-06-05, same audit standard):

- **`lean/`** — Qwen-family robustness study (3 sizes x 3 quants x 3 contexts = 27
  triples) with a single-objective MILP under hard memory + latency SLA and a
  per-group **concurrent (batched) latency model**. Canonical `lean_8gb` run:
  Q = 1.881 at T° = 8 s, memory 5.43 GB, dispatcher Qwen-3B Q8 + a shared
  Qwen-1.5B Q5 group serving specialist and synthesiser. Concurrent serving
  dominates the sequential model everywhere and stays feasible at T° = 4-6 s where
  sequential rejects the instance. Deterministic by construction (HiGHS,
  seed=0, 1 thread); catalog carries a SHA-256 sidecar; 65 tests.
- **`part4_dynamic_path/`** — a training-free, LLM-free **dynamic agentic-path**
  layer on the parts-2/3 static optimum: per-query specialist pruning via
  retrieval-geometry signals with a conformal coverage floor + cost-aware
  submodular coverage. The shipped `results/dynamic_eval.json` is the
  **hashing-proxy** artifact (internally verified); the paper's bge-m3 numbers
  require the one-step score-cache build on the full-codebase machine
  (`part4_dynamic_path/data/INPUTS.md`). |
| **part3_bilinear_gated** | Do the roles interact? | maximize **gated** quality: specialist contribution × the router's measured recall on that domain | **Yes.** A specialist is wasted if the router misroutes it. Modelled as a **bilinear** term (router × specialist), linearized exactly with McCormick. The gated optimum differs from the additive one at nearly every budget and selects a small, high-recall router that global-F1 selection misses. |

The intellectual point of the paper: **a parameter proxy and a measured-quality,
coupling-aware objective induce materially different optimal allocations.** Cost
and capability are not the same axis, and in a multi-agent pipeline the roles do
not decouple.

## 3. Repository layout

```
mamap_repo/
├── README.md                  ← this file
├── RUNBOOK.md                 ← exact end-to-end reproduction order + runtimes
├── requirements.txt
├── shared/                    ← inputs used by all three parts (see shared/README.md)
│   ├── data/                  ← agents.yaml, catalog_zoo.yaml, calibration, perf & quality tables
│   ├── corpus/                ← Italian law + case-law manifests
│   ├── faiss_code/            ← the real retrieval/indexing pipeline (chunking, bge-m3, FAISS)
│   ├── pareto/                ← the three precomputed Pareto frontiers (regenerable)
│   └── lib/paths.py           ← central repo-relative paths (no absolute paths anywhere)
├── part1_static_allocation/   ← parameter-proxy MILP, baseline, performance figures
│   ├── src/                   ← 01..09 scripts
│   └── results/{figures,data} ← outputs land here
├── part2_quality_aware/       ← measured-quality MILP + analysis
│   ├── src/                   ← 00 (prepare) + 01..06
│   └── results/{figures,data}
├── part3_bilinear_gated/      ← gated (bilinear) MILP + frontier
│   ├── src/                   ← 01..03
│   └── results/{figures,data}
└── paper/                     ← the LaTeX source and the figures it includes
```

Each of `shared/`, `part1`, `part2`, `part3`, `paper/` has its **own README and
RUNBOOK**. Start with this file, then `RUNBOOK.md`.

## 4. Quick start

```bash
pip install -r requirements.txt
# then follow RUNBOOK.md — in short:
cd part1_static_allocation/src && python3 01_build_capacity_frontier.py && ...
cd ../../part2_quality_aware/src && python3 00_prepare_intermediates.py && ...
cd ../../part3_bilinear_gated/src && python3 01_gated_milp_headtohead.py && ...
```

All scripts run from inside their own `src/` directory and write to their part's
`results/`. They locate inputs through `shared/lib/paths.py`, so the package works
unchanged after unzip or `git clone`.

## 5. Honest scope (read this)

These are stated in the paper and repeated here so results are not over-read:

- **`capacity` is a parameter-count proxy**, not measured answer quality. Part 1 is
  explicit about this; Part 2 is the measured rebuttal.
- **The heterogeneous affinities in part 1 (script 03/09) are *synthetic*** and
  labelled as such; they demonstrate *when* optimization is needed, not a measured
  result. The measured heterogeneity is in Part 2.
- **Quality is measured judge-free** (embedding-based RAGAS): correctness =
  cos(answer, gold), faithfulness = max cos(answer, context), etc. It is **not**
  validated against human legal experts.
- **Quality is scored under oracle retrieval** (gold passages supplied as context),
  so it isolates *generation* quality from *retrieval* quality. The live FAISS
  retriever is reported separately as a diagnostic.
- **The evaluation set is small** (37 root queries; 7 of 9 domains have n < 10).
  Per-domain claims carry bootstrap CIs and are labelled indicative; pooled and
  cross-role claims are the robust ones.
- **No held-out train/test split**: the quality-aware and gated frontiers are
  in-sample optima. The per-domain routing recall in Part 3 uses the same small
  samples, so its *specific* winning router is indicative; the structural claims
  (coupling changes the optimum; recall-on-activated-domains is the right quantity)
  are robust.

## 6. Hardware note

All measurements (`shared/data/perf_table.parquet`) come from one NVIDIA RTX 4070
Laptop GPU (8 GB, ~6.99 GB usable) via `llama.cpp`. The optimization and analysis
do not require a GPU — they run on CPU from the released tables in minutes. Only
re-measuring the performance/quality tables from scratch needs the GPU and the
model weights.

## 7. Determinism and exact reproduction

Reproducibility is stated precisely, in two layers:

- **Objective values and verdicts are machine-independent.** Each MILP is solved to
  proven optimality (the bilinear part-3 solves use `timeLimit=60 s`; the hardest
  stage-1 instance proves optimality in ~48 s on the reference laptop). The paper's
  headline numbers — **44 gated points, bilinear > additive at 42/44, mean gap 0.18,
  max 0.32**, the 7/132 and 100/132 baseline counts, every frontier objective — are
  the same on any machine.
- **Reported allocations are made machine-independent by canonical solution
  selection.** A MILP's optimal *solution* can be a set (e.g. two quantizations of
  the same model with equal objective); which member a solver returns depends on
  the solver build. We therefore refine every reported solution lexicographically
  over the optimal face — objective, then minimum system latency, then minimum
  loaded memory (`shared/lib/lexsolve.py`) — and pin the solver build
  (`pulp==3.3.2`). Either mechanism alone suffices for byte-identical CSVs;
  together they are belt-and-braces.
- Bootstrap confidence intervals (part 2, q12) use a fixed seed; the synthetic
  affinities (part 1/03) use a fixed seed.
- **`closeout.ps1`** (repo root) packages the author-side close-out: Phase A runs
  the full pipeline, `verify.py`, and a byte-identity comparison of all
  deterministic artifacts against `replication/canonical_hashes.sha256`;
  Phase B builds the bge-m3 score cache and produces part 4's paper-grade
  results. Status of every residual trust item: `replication/CLOSEOUT_REPORT.md`;
  every change made during the audit, as unified diffs:
  `replication/AUTHOR_REVIEW_DIFFS.md`.
- **`python3 verify.py`** (repo root) re-checks every headline paper number against
  the regenerated artifacts and exits non-zero on any mismatch. Run it after the
  RUNBOOK pipeline: it is the normative paper-number -> artifact mapping.
- The longest step is `part3/02` (exact 44-point sweep; ~3 lexicographic stages per
  point); it writes its CSV incrementally so progress is visible and survives
  interruption. Full repository wall-clock on a laptop CPU is ~25–40 min
  (see `RUNBOOK.md`).
