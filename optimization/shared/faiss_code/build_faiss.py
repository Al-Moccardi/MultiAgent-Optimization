"""
part1_allocation/tools/build_faiss.py
======================================
Build FAISS indexes for the law and case corpora using a chosen embedding model.

Input  : the gold-text yaml (calib_clean_with_gold_text.yaml), which carries the
         passage TEXT in gold_law_passages / gold_case_passages, OR a corpus jsonl.
Output : <out>/LawCorpus_IT.faiss      + LawCorpus_IT.manifest.json
         <out>/CaseCorpus_IT.faiss     + CaseCorpus_IT.manifest.json
         <out>/corpus_text.jsonl       (merged id -> text store, for retrieval)

Embedder: default BAAI/bge-m3 (multilingual, 1024-dim). Use --embedder hash for a
dependency-free offline build (validates the pipeline without a model download).

Examples
--------
# real, recommended (needs sentence-transformers + a one-time bge-m3 download):
python -m part1_allocation.tools.build_faiss \
    --gold-text part1_allocation/data/calib_clean_with_gold_text.yaml \
    --embedder bge-m3 --out shared/corpus

# offline (no download), to verify the plumbing:
python -m part1_allocation.tools.build_faiss \
    --gold-text part1_allocation/data/calib_clean_with_gold_text.yaml \
    --embedder hash --out shared/corpus
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from part1_allocation.scoring.corpus import split_corpus_from_gold_text
from part1_allocation.scoring.embeddings import make_embedder
from part1_allocation.scoring.retrieval import build_index


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build law/case FAISS from gold-text passages")
    ap.add_argument("--gold-text", required=True, help="calib_clean_with_gold_text.yaml")
    ap.add_argument("--embedder", default="bge-m3",
                    help="embedding model: bge-m3 | e5 | <hf-id> | hash (offline)")
    ap.add_argument("--embed-device", choices=["auto","cpu","cuda"], default="auto",
                    help="run the embedder on cpu to spare VRAM for other models")
    ap.add_argument("--out", default="shared/corpus", help="output directory")
    args = ap.parse_args(argv)

    data = yaml.safe_load(Path(args.gold_text).read_text(encoding="utf-8"))
    law, case = split_corpus_from_gold_text(data["queries"])
    print(f"[build_faiss] law passages: {len(law)} | case passages: {len(case)}")

    embedder = make_embedder(args.embedder,
                             device=None if args.embed_device=="auto" else args.embed_device)
    print(f"[build_faiss] embedder dim={embedder.dim} "
          f"({type(embedder).__name__})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_law = build_index(law, embedder, out / "LawCorpus_IT.faiss",
                        out / "LawCorpus_IT.manifest.json")
    n_case = build_index(case, embedder, out / "CaseCorpus_IT.faiss",
                         out / "CaseCorpus_IT.manifest.json")

    # merged id->text store for the retriever / oracle contexts
    merged = {**law, **case}
    with open(out / "corpus_text.jsonl", "w", encoding="utf-8") as f:
        for k, v in merged.items():
            f.write(json.dumps({"id": k, "text": v}, ensure_ascii=False) + "\n")

    print(f"[build_faiss] wrote {n_law} law + {n_case} case vectors -> {out}/")
    print(f"[build_faiss] corpus_text.jsonl: {len(merged)} docs")


if __name__ == "__main__":
    main()
