# shared/ — inputs and code used by all three parts

This folder holds everything common to part1/part2/part3: the canonical input data,
the legal corpora, the real retrieval pipeline, the precomputed Pareto frontiers,
and the central path library. **Nothing here is part-specific; the three parts read
from here and write only to their own `results/`.**

## shared/data/ — canonical inputs (read-only)

| File | What it is |
|------|-----------|
| `agents.yaml` | The 11 agents: 1 dispatcher, 9 domain specialists, 1 synthesiser, each with its minimum context `c_min`. Defines the pipeline. |
| `catalog_zoo.yaml` | The 11 candidate models (HF ids, params, quantizations, contexts). Expands to **84 configurations**. |
| `calibration_with_gold.yaml` | The **25 evaluation queries**: each with gold routing (expected specialists), difficulty, risk, and gold supporting authorities (statute + case ids **and** passage texts). |
| `perf_table.parquet` | Measured cost for all 84 configs on the RTX 4070: peak VRAM, TTFT, throughput, energy/token. The hardware-dependent input. |
| `quality_table.parquet` | Per-(agent, config, query) measured quality records (dispatcher routing + specialist/synth RAG). The raw quality campaign output. |
| `quality_scorecard.csv` | Per-(agent, config) aggregated quality + RAGAS components. The convenient form used by most analysis. |
| `retriever_diagnostic.csv` | Live-FAISS retrieval context precision/recall per query (separate from the oracle-retrieval quality scores). |

## shared/corpus/ — legal corpora manifests

`LawCorpus_IT_manifest.json` (63 statute units: 60 Codice Civile, 2 CPC, 1 D.Lgs
28/2010) and `CaseCorpus_IT_manifest.json` (61 items: 57 ECLI decisions, 3 mediation
records, 1 notarial act). These are the id lists indexed by FAISS. The passage
*texts* live in the calibration file's gold passages.

## shared/faiss_code/ — the real retrieval pipeline

The exact indexing/retrieval code described in the paper's Experimental Setting:

- `corpus.py` — the id→text store and the gold-text → (law, case) split. Also
  implements the **oracle-retrieval** context resolution used by the quality scoring.
- `embeddings.py` — `BAAI/bge-m3` (multilingual, 1024-d, L2-normalized) with a
  dependency-free hashing fallback for offline plumbing checks.
- `retrieval.py` — builds a **FAISS `IndexFlatIP`** (exact inner-product = cosine)
  per corpus + an id manifest; the live retriever embeds the query, searches both
  indexes, merges and dedups, returns top-k.
- `scorer.py` — the quality metrics: routing F1 (multi-label set), correctness =
  cos(answer, gold), faithfulness = max cos(answer, context), answer relevancy;
  aggregate Q = 0.5·correctness + 0.5·mean(faithfulness, relevancy); context
  precision/recall excluded from Q (diagnostics only); synth Q = correctness.
- `build_faiss.py` — CLI that builds the indexes from the gold-text file.

**Chunking** is at legal-unit granularity: one statute article / one case passage =
one document, no sub-article windowing or overlap (civil-code articles are already
short self-contained norms).

You do **not** need to run this code to reproduce any paper result — the perf and
quality tables are the released inputs. It is here so the retrieval setup is fully
specified and rebuildable. To rebuild the indexes (needs `sentence-transformers`,
`faiss-cpu`):
```bash
cd shared/faiss_code
python3 build_faiss.py --gold-text ../data/calibration_with_gold.yaml --embedder bge-m3 --out ../corpus
# or --embedder hash for a no-download offline build
```

## shared/pareto/ — precomputed Pareto frontiers (regenerable)

| File | Produced by | Meaning |
|------|-------------|---------|
| `perf_capacity_frontier.csv` | part1/01 | parameter-capacity vs latency frontier (the proxy objective) |
| `quality_additive_frontier.csv` | part2/01 | measured additive pipeline quality vs latency |
| `quality_gated_frontier.csv` | part3/02 | gated (bilinear) pipeline quality vs latency |

These ship precomputed so the plotting scripts work immediately. The three "01/01/02"
scripts regenerate them in place — see each part's RUNBOOK.

## shared/lib/paths.py — central paths

Every script does `import paths as _P` (via a 3-line header) and refers to
`_P.PERF_TABLE`, `_P.SCORECARD`, `_P.PARAMS_B`, `_P.MEM_BUDGET_GB`, etc. There are
**no absolute paths** in the repository; `paths.py` resolves everything relative to
the repo root, so the package runs unchanged after unzip or clone.
