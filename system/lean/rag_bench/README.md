# `rag_bench/` — domain-labelled RAG benchmark harness

This harness lets you score a `Catalog` of (model, quant, context) configs on
**any domain-labelled RAG benchmark** so the lean MILP has real per-(role,
config, domain) quality numbers.

The colleague's parts/1–3 score on an in-house 37-query Italian family-law
set with 9 domain specialists. The lean version *targets the same shape*
(specialist = **domain expert**) but stays generic about the underlying
dataset, since "domain" is a topic label, not a hop type.

## Input schema

A single JSONL file with one record per query:

```jsonc
{
  "query_id": "med_001",
  "domain": "medical",          // free-form domain tag (string)
  "query": "What are the side effects of metformin?",
  "gold_context": "Metformin commonly causes ...",   // oracle retrieval
  "gold_answer": "Common side effects include nausea, ..."
}
```

`build_subset.py` reads this, validates, stratified-samples per domain, and
writes `lean/rag_bench/subset.parquet`. `eval.py` then iterates `(role,
config_id, query)`, generates an answer with a pluggable `Generator`, scores
it via the colleague's RAGAS-aggregate (`shared/faiss_code/scorer.py`), and
writes `lean/catalog/quality.parquet` consumed by the MILP.

## Recommended datasets

Pick **one** of:

1. **CRAG** (Meta, 2024) — five built-in domains (finance / sports / music /
   movies / open), public on Hugging Face, has gold answers and retrieval
   contexts. Cleanest single-dataset fit. `dataset_id =
   "Microsoft/Cosmos-CRAG"`.
2. **Curated 4-BEIR mix** — slice four BEIR datasets into a multi-domain
   set: `nfcorpus` (medical), `fiqa-2018` (finance), `scifact` (scientific),
   `arguana` (argument-mining / general). All BEIR rows already carry
   queries + gold passages.
3. **PubMedQA + FiQA + SciFact + HotpotQA** — three single-domain RAG sets
   plus HotpotQA as the "general" bucket. Most plumbing per record.

Each comes with its own licence; check before redistributing.

## Quickstart (with a JSONL you already have)

```bash
# 1. Build the stratified subset (cached parquet).
python -m rag_bench.build_subset --input my_dataset.jsonl --out subset.parquet

# 2. Score it — defaults to the MockGenerator for CI; pass --real to use a
#    locally hosted llama.cpp server, or --judge "openai:gpt-4o-mini" to use
#    an external judge instead of the embedding-based RAGAS aggregate.
python -m rag_bench.eval --subset subset.parquet --out ../catalog/quality.parquet
```

## Status

This package ships the **harness**: data schema, build script, eval skeleton
with a `MockGenerator` (analytical fallback) and a `Generator` Protocol so
real backends (vLLM, llama.cpp server, OpenAI-compatible) plug in cleanly.

The actual benchmark run is *out of scope for the lean MVP*. The lean MILP
test suite uses `src.quality.synthetic_quality(...)` until real
`quality.parquet` lands.
