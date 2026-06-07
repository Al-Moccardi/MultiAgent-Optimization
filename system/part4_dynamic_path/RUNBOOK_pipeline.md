# RUNBOOK — end-to-end test: normal vs dynamic routing (held-out queries)

`run_pipeline.py` runs the **real** pipeline on the **58 non-overlapping** test
queries (those not in the calibration set used to build the gate), twice per
query, and reports latency / Δlatency and correctness / Δcorrectness.

This runs in the **full codebase** (the one with `part1_allocation/inference/`,
the GGUF models, and `shared/corpus/`), not the data-only replication repo.

## What it compares
- **NORMAL routing** — the dispatcher LLM routes the query; *all* its predicted
  specialists run (RAG retrieve + generate); the synthesiser composes the answer.
- **DYNAMIC routing** — same dispatcher prediction, then the Part-4 gate prunes
  it to a per-query subset S(q) (or abstains); only those specialists run.

Both passes share the same dispatcher candidate set, so the difference is purely
the adaptive gate — that is what Δlatency and Δcorrectness isolate.

## Correctness on held-out queries (they carry NO gold answer)
- **Out-of-domain** queries (`expected_outcome: abstain`): correctness = correct
  abstention (1 if the pipeline abstained, else 0). Reference-free.
- **In-domain** queries (no gold answer exists): correctness = judge-free
  RAGAS-style **answer quality** with bge-m3 (faithfulness to retrieved law +
  answer relevancy). Reported as `answer_quality` — *not* gold-correctness.
- Optional `--judge_model`: score in-domain correctness with an LLM judge instead
  (gold-free), if you have one. Off by default.

> If you want true gold-correctness on the held-out set, the only honest way is to
> add `ground_truth_answer` for those queries (expert or strong-reference) — they
> don't have one today, so the pipeline uses the gold-free metric above.

## Requirements
- `llama-cpp-python` built for your hardware, and the GGUF model files at the
  `gguf_path`s in `catalog_zoo.yaml` (the same models that produced the tables).
- `sentence-transformers` + the bge-m3 weights (retrieval signal, RAG, and the
  gold-free quality metric).
- `shared/corpus/` with `corpus_text.jsonl` + the `*.faiss` indexes (build once
  with `build_faiss --embedder bge-m3` if absent).

## Run
```bash
# from the FULL-codebase repo root
python -m part4_dynamic_path.run_pipeline --config gated_optimum --alpha 0.10
python -m part4_dynamic_path.run_pipeline --config qa_optimum
python -m part4_dynamic_path.run_pipeline --config qwen

# smoke test (no GGUF / GPU needed) — verifies the flow, NOT the numbers:
python -m part4_dynamic_path.run_pipeline --config gated_optimum --mock --limit 8
```

Flags:
- `--config {qa_optimum,gated_optimum,qwen}` — the fixed allocation (per-role models).
- `--router {llm,free,all}` — candidate source for BOTH passes: the real
  dispatcher LLM (default), the free LLM-free retrieval router, or all 9 domains.
- `--mode {concurrent,sequential}` — specialist-stage latency model (concurrent =
  §10.5 batched, the default; sequential = sum of specialist calls).
- `--quant_disp/--quant_spec/--quant_synth` — quantization per role (defaults
  Q5_K_M / Q8_0 / Q5_K_M).
- `--limit N` — cap the number of queries (0 = all 58).

## Output  (`results/data/pipeline_eval.json`)
`summary`:
- `latency`: `normal_mean_s`, `dynamic_mean_s`, `delta_mean_s`, `delta_pct`.
- `correctness_indomain_answer_quality`: `normal`, `dynamic`, `delta` (gold-free).
- `abstention_accuracy`: `normal`, `dynamic`, `delta` (out-of-domain).
- `mean_k`: normal vs dynamic activated-specialist counts.

`per_query`: for each query, both passes' `S`, `k`, `latency_s`, `correctness`,
the (truncated) synthesised `answer`, and `delta_latency_s` / `delta_correctness`.

## Reading it
Expect dynamic routing to **cut latency** (fewer specialists; abstains on
out-of-domain) and hold or slightly improve **answer quality** (it drops
irrelevant specialists that add noise to the synthesiser). The honest headline is
"normal vs dynamic at equal answer quality, with X% latency reduction and correct
abstention on the out-of-domain set." A *negative* Δcorrectness on in-domain
queries would mean the gate is over-pruning — check those per-query rows (the
`dynamic.S` will be missing a gold domain).

## Honest notes
- `--mock`/proxy is a plumbing check only; numbers require real models + bge-m3.
- In-domain "correctness" here is gold-free answer quality, not correctness vs a
  reference answer — state that precisely in the paper.
- The conformal coverage guarantee is marginal at ~25–37 calibration queries.
