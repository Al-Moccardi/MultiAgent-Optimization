# MAMAP-Edge — applied patches (Tier 1 + Tier 2 + Tier 3)

This codebase is the original `mamap-edge` with three reviewed changesets applied.
With the sole exception of the Tier-1 bug fix, **every change is additive**: all
defaults reproduce the original v3 behaviour, so nothing changes unless you opt in.

Test status: **37 passed**. The one failing test
(`test_faiss_build_and_retrieve_roundtrip`) is a pre-existing environment skip —
it needs `faiss` installed — and is unrelated to these changes.

---

## Tier 1 — bug fix (changes default behaviour, because the old behaviour was wrong)
- **`optimize/pareto.py`** — `latency_grid` now builds the eps grid on the SYSTEM
  latency scale (L_router + L_max_spec + L_synth) that constraint (8) actually
  bounds, instead of a single agent's latency range. The old grid sat below the
  feasible system latency once specialists+synth were present, collapsing the
  sweep onto the unconstrained point and hiding the entire latency frontier.
- **`tests/test_pareto_grid.py`** — regression tests.

## Tier 2 — opt-in modelling options (defaults unchanged)
- **Finding 3 — correct expected-latency model.** `--latency-model {worst_case,
  expected_max}`. expected_max is the architecturally-correct expected latency
  for PARALLEL specialists (mean over queries of the slowest ACTIVE specialist);
  the Sigma_s pi_s L_s sketched in v3 9.1 is a serial-work proxy and is
  deliberately NOT used. Needs per-query gold active sets, now emitted by derive.py.
- **Finding 4 — objective weighting.** `--normalize-specialists` and
  `--obj-weights w_rt,w_syn,w_spec`.
- Files: optimize/milp.py, optimize/derive.py, optimize/pareto.py,
  pipeline/run_all.py, tests/test_tier2_options.py.

## Tier 3 — the two items that legitimately become code (defaults unchanged)
Tier 3 is mostly dataset / measurement / paper work, not code. Two items had a
real code deliverable; the other two do not (see below).

- **Finding 6 — decoupled correctness embedder.** `--corr-embedder <name>`. By
  default correctness reuses the retrieval embedder (original behaviour), which
  lets a model that parrots retrieved text score high on both faithfulness and
  correctness. Passing a DIFFERENT embedder scores correctness with a model
  independent of the retriever, breaking that circularity. Relevancy and
  faithfulness still use the main embedder.
  Files: scoring/scorer.py, factories.py, pipeline/run_all.py.
- **Finding 8 — optimum-robustness bootstrap.** A new module + CLI that resamples
  the measured queries (with replacement), rederives Q3 and re-solves the MILP,
  and reports objective confidence intervals + per-agent config selection
  frequency. Pure post-processing on the parquets — no model inference.
  Files: optimize/bootstrap.py, tools/bootstrap_report.py, tests/test_tier3_options.py.

      python -m part1_allocation.tools.bootstrap_report \
          --bundle shared/pareto --n-boot 300 --latency-model worst_case

### NOT changed in code (and why)
- **Finding 5 — specialist Q excludes correctness.** The remedy is per-specialist
  ground truth — a dataset authoring task (writing legal gold answers per
  specialist domain). The code already supports including correctness
  (include_correctness=True); the missing piece is data only a domain expert can
  write. No code can substitute for it.
- **Finding 7 — cosine-metric weakness.** The remedy is paper caveats plus
  validation against a human-scored subset. Not a code change.

Both are documented in MAMAP_Tier2_audit.md.

---

## New CLI flags (run_all.py)
    --latency-model {worst_case,expected_max}   # default worst_case   (Finding 3)
    --normalize-specialists                      # default off          (Finding 4)
    --obj-weights w_rt,w_syn,w_spec              # default 1,1,1        (Finding 4)
    --corr-embedder <name>                       # default: reuse --embedder (Finding 6)

## Suggested ablations for the paper
    # baseline (v3 as-is)
    python -m part1_allocation.pipeline.run_all --mode real --device auto --gold-text ...
    # expected-latency variant
    ... --latency-model expected_max
    # normalised objective
    ... --normalize-specialists
    # correctness scored independently of the retriever
    ... --corr-embedder e5
    # robustness of the chosen allocation
    python -m part1_allocation.tools.bootstrap_report --bundle shared/pareto --n-boot 300

## What was NOT changed across all tiers, and why
- Worst-case latency stays the default (defended v3 9.1 choice; expected_max is added).
- Q3 normalisation is untouched — the earlier "double-count" concern was withdrawn
  after analysis (Q3 is a legitimate joint expectation).
- Validity items requiring data or a human study (Findings 5, 7) are written up,
  not faked in code.

## Campaign speed flags (added; defaults preserve current behaviour)
Three flags reduce real-campaign wall-clock without changing any default:

- `--no-logprobs` : disables `logits_all=True` in the backend. This is the
  single biggest speedup (2-5x) because logits_all forces per-position logit
  computation. It only drops the confidence signal, which is used solely by the
  (unimplemented) Stage-2 cascade -- Stage-1 allocation does not use it.
- `--max-tokens N` (default 512) : caps output tokens in the quality sweep.
  Generation time scales with this. Lowering to ~256 roughly halves generation
  time; keep it large enough that answers are not truncated (derived latency
  L = ttft + n_tok/throughput reflects the measured output length).
- `--perf-repeats N` (default 3) : timed repetitions per config in the perf
  sweep (median taken). Lower to 1-2 to speed the cheap phase.

Recommended fast-but-faithful campaign invocation:
    python -m part1_allocation.pipeline.run_all --mode real --device auto \
        --latency-sla 10 --catalog .../catalog.zoo.yaml \
        --gold-text .../calib_clean_with_gold_text.yaml \
        --embedder bge-m3 --embed-device cpu --retrieve \
        --no-logprobs --max-tokens 384

## Synthesiser: real specialist answers + correctness-only quality
Two changes to how the synthesiser is measured (specialists unchanged):

1. **Input = real specialist answers (not idealised contexts).** The quality
   sweep is now two-pass. Pass 1 measures every agent except the synth and, for
   each query, records each specialist's answer at a fixed REFERENCE config (the
   largest-context group -> reproducible, no circularity with the MILP choice).
   Pass 2 feeds the synth the COMPOSED real answers of the GOLD-activated
   specialists, each capped to an EQUAL share of the input budget
   (budget / n_activated) so every specialist contributes evenly, and scores the
   synth against the gold answer. This replaces the previous "Mode (a)"
   approximation that fed the synth the gold/retrieved contexts directly.
   The MILP is unchanged: the synth stays one record per config (linear term).
   The router-config dependence of the synth is intentionally not modelled
   (would require a second bilinear block); the gold active set is used instead.

2. **Synth quality Q = correctness alone.** `SpecialistScorer(correctness_only=True)`
   for the synth: Q = cos(answer, gold_answer), with faithfulness/relevancy still
   measured and saved but excluded from Q. Rationale: the synth produces the
   final user-visible answer from the real specialist outputs, so correctness vs
   the gold already reflects every upstream error; mixing in grounding dilutes
   that signal. Specialists are unchanged (Q = mean(faithfulness, relevancy),
   correctness excluded, since their gold is the full composed answer).

Files: measure/quality.py (two-pass + capping helpers), scoring/scorer.py
(correctness_only), factories.py (synth uses correctness_only).

## Crash resilience for long campaigns (added)
Three protections so a multi-hour real run survives a GPU crash:

- `--reuse-perf <perf_table.parquet>` : skip the performance sweep and load an
  existing perf table (per-device; reuse only one measured on THIS hardware). It
  is filtered to the current --catalog automatically. Mirrors --reuse-quality.
- **Incremental quality checkpoint** : the quality sweep now writes
  `quality_table.partial.parquet` to the output dir after EVERY (model,quant)
  group. A crash no longer loses completed generations; resume with
  `--reuse-quality <...>/quality_table.partial.parquet`.
- **Skip-on-generation-failure** : a model that OOMs or errors mid-generation
  (e.g. an oversize config exceeding VRAM) is skipped with a warning instead of
  aborting the whole sweep. Records from earlier groups are preserved.

### Resuming a crashed run (typical)
    # perf_table.parquet already exists from the first run; reuse it + partial quality:
    python -m part1_allocation.pipeline.run_all --mode real --device auto \
        --latency-sla 10 --catalog part1_allocation/config/catalog.zoo.yaml \
        --gold-text part1_allocation/data/calib_clean_with_gold_text.yaml \
        --embedder bge-m3 --embed-device cpu --retrieve --no-logprobs --max-tokens 384 \
        --reuse-perf shared/pareto/perf_table.parquet \
        --out shared/pareto

NOTE on the CUDA crash seen in practice: configs whose peak_mem_gb exceeds the
GPU's physical VRAM (e.g. mistral-7b Q8_0 ~7.5GB on an 8GB card) can OOM. Either
remove them from the catalogue, or rely on the new skip-on-failure to step past
them (they would be dropped by the MILP budget anyway).
