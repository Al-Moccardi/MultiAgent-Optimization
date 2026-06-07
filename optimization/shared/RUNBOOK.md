# shared/ RUNBOOK

`shared/` is **not run as a pipeline** — it holds inputs, the retrieval code, and
the regenerable frontiers. There are only two optional operations here.

## A. Verify the inputs load (sanity check)

```bash
cd shared
python3 - <<'PY'
import sys; sys.path.insert(0,"lib")
import paths as P, pandas as pd, yaml, json
print("agents:", len(yaml.safe_load(open(P.AGENTS))["agents"]))
print("models:", len(yaml.safe_load(open(P.CATALOG))["models"]))
print("queries:", len(yaml.safe_load(open(P.CALIBRATION))["queries"]))
print("perf rows:", len(pd.read_parquet(P.PERF_TABLE)))
print("quality rows:", len(pd.read_parquet(P.QUALITY_TABLE)))
print("scorecard rows:", len(pd.read_csv(P.SCORECARD)))
print("law units:", len(json.load(open(P.LAW_MANIFEST))),
      "| case units:", len(json.load(open(P.CASE_MANIFEST))))
PY
```
Expected: 11 agents, 11 models, 25 queries, 84 perf rows, 9016 quality rows, 560
scorecard rows, 63 law + 61 case units.

## B. (Optional) Rebuild the FAISS indexes

Not required for any paper result. Needs `sentence-transformers` + `faiss-cpu`
(and a one-time bge-m3 download), or use the offline hashing embedder.

```bash
cd shared/faiss_code
# real (recommended):
python3 build_faiss.py --gold-text ../data/calibration_with_gold.yaml --embedder bge-m3 --out ../corpus
# offline plumbing check (no download):
python3 build_faiss.py --gold-text ../data/calibration_with_gold.yaml --embedder hash --out ../corpus
```
This writes `LawCorpus_IT.faiss`, `CaseCorpus_IT.faiss`, their manifests, and
`corpus_text.jsonl` into `shared/corpus/`.

## C. The Pareto frontiers

`shared/pareto/*.csv` ship precomputed. They are **regenerated** by:
- `part1/src/01_build_capacity_frontier.py`  → `perf_capacity_frontier.csv`
- `part2/src/01_quality_aware_milp.py`       → `quality_additive_frontier.csv`
- `part3/src/02_gated_frontier_and_comparison.py` → `quality_gated_frontier.csv`

You do not run anything in `shared/` to refresh them; running the relevant part does.
