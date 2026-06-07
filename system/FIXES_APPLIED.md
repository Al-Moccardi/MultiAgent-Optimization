# Fixes applied to the MAMAP codebase

Two reviewer-flagged issues addressed. All changes are ADDITIVE and non-breaking:
the full test suite still passes (part1: 42 passed — 38 original + 4 new; lean: 27 passed).

## Fix 1 — Latency model reconciliation (paper §6.2 Eq. 7 vs code)

PROBLEM: the paper's headline chain is sequential, `L_disp + k·L_spec + L_synth`,
but the released MILP only implemented `worst_case` (`L_disp + max_s(L_s) + L_synth`),
which — because specialists share one model — collapses to a SINGLE specialist call
(the concurrent model). Headline figure and solver disagreed.

FIX: added a third, selectable latency model `"sequential"` implementing the paper's
Eq. (7) exactly.
  - `part1_allocation/optimize/milp.py`: new `latency_model="sequential"` + `k_activated`
    param. Constraint becomes `L_router + k_activated·L_spec_one + L_synth ≤ ε`, where
    `L_spec_one ≥ L_s ∀ specialists` (shared-config slot latency). Reported `system_latency`
    is consistent with the chosen model.
  - `part1_allocation/optimize/pareto.py`: `latency_grid` now spans the correct scale
    for sequential (×k); `build_frontier` threads `latency_model`/`k_activated` through.
  - `part1_allocation/pipeline/run_all.py`: new `--latency-model sequential --k-activated K`.
  - `part1_allocation/tests/test_sequential_latency.py`: 4 new tests pin the arithmetic
    (concurrent 5.4s; sequential k=3 → 9.4s; infeasible at the concurrent budget; k=1 ≡ worst_case).

REPRODUCE the paper's Fig. 7 / Table 3 (sweep k ∈ {1,3,5,9}):
    python -m part1_allocation.pipeline.run_all --perf-table shared/pareto/perf_table.parquet \
        --latency-model sequential --k-activated 3
  (the default remains `worst_case`, so nothing else changes).

NOTE on choice: the paper now has BOTH models — sequential is the headline (§6–7),
concurrent (=worst_case here) is §10.5. They are both correct models of different
execution regimes; the paper reports the optimum is invariant between them.

## Fix 2 — Ship the data so routing_eval reproduces offline

ADDED, byte-identical to the originals (MD5-verified, not modified):
  - `data/manifests/calibration_queries_en.yaml` — the REAL 94-query routing-annotated
    calibration set (expected_agents only, no gold passages). md5 1f5d92987d21e9adb8402339bc0df17b.
    Verified: 94 ∩ 64-test = 0 (no leakage); the paper's key claim holds.
  - `shared/pareto/perf_table.parquet` — RESTORED the REAL RTX 4070 table (84 rows,
    hardware="DESKTOP-OM76TBD__NVIDIA GeForce RTX 4070 Laptop GPU", energy on all 84).
    md5 10909da342119edbbd7232d0f3f82fd4. (The repo had a 20-row macbook SYNTHETIC
    stand-in from an earlier sandbox run; that is now replaced by the genuine table.)

PATH WIRING:
  - `shared/lib/paths.py`: new `CALIBRATION_ROUTING` -> data/manifests/calibration_queries_en.yaml
    (falls back to the 25-query gold-text file if absent).
  - `part4_dynamic_path/routing_eval.py`: now DEFAULTS to the 94-query file (with fallback),
    so `python -m part4_dynamic_path.routing_eval --beta 3` reproduces the paper's setup
    (64 held-out, calib=94, no leakage) with no flags.

### The bge-m3 dependency (read carefully)

The HEADLINE numbers (routing F1 0.33→0.50, OOD abstention 0.60) require the bge-m3
embedder. Without it, the code falls back to a hashing PROXY that gives WEAKER numbers
(F1 ≈ 0.29→0.32, abstention 0.00) and prints `NOTE: PROXY embedder`. The proxy result
is NOT the paper's result and must never be reported as such.

To reproduce the paper's bge-m3 numbers on a machine WITHOUT bge-m3 / without a GPU,
a FROZEN score-cache mechanism is provided:
  - `part4_dynamic_path/scripts/build_score_cache.py` — run ONCE on a bge-m3 machine:
        python -m part4_dynamic_path.scripts.build_score_cache --embedder bge-m3
    It computes REAL per-(query,domain) relevance scores and writes
    `part4_dynamic_path/data/score_cache.json`.
  - `part4_dynamic_path/routing_eval.py` auto-loads that cache if present and REPLAYS
    the exact scores (no embedder call), so F1=0.50 / abstention=0.60 reproduce offline.

IMPORTANT / HONESTY: this repository does NOT ship a pre-populated score_cache.json,
because the only correct way to populate it is from the real bge-m3 embedder on your
hardware. No relevance scores have been fabricated or hand-written. Until you run
build_score_cache.py with --embedder bge-m3, offline runs use the proxy and will show
the weaker proxy numbers (clearly labelled).

### Data-provenance note for the paper (overlaps, stated honestly)
Verified on the shipped files:
  - 94-calib ∩ 64-test  = 0   (no train/test leakage — the central claim)
  - 25-profile ∩ 64-test = 6  (the relevance PROFILES share 6 queries with the test set;
    a mild optimistic bias on the relevance signal — worth disclosing in Limitations)
  - 94-calib ∩ 25-profile = 9 (threshold-fitting and profiling share 9 queries; different
    operations — routing-label fitting vs passage-profile building)
