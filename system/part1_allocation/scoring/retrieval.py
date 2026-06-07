"""
part1_allocation/scoring/retrieval.py
=====================================
Build and query FAISS indexes over the corpus, using any `Embedder`.

A FAISS index stores only vectors, so we also write a **manifest** (the list of
doc ids in index-add order) to map a search hit back to its id -> text. This is
the piece the project's original `.faiss` files were missing.

Two indexes (law, case) mirror the gold-set split. Retrieval embeds the query,
searches both, and returns the top-k passage texts as RAG contexts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .embeddings import Embedder


def build_index(corpus: dict[str, str], embedder: Embedder,
                out_index: str | Path, out_manifest: str | Path) -> int:
    """Embed every doc and write an IndexFlatIP + an id manifest. Returns #docs."""
    import faiss
    ids = list(corpus.keys())
    if not ids:
        raise ValueError("empty corpus")
    vecs = embedder.encode([corpus[i] for i in ids])
    index = faiss.IndexFlatIP(embedder.dim)
    index.add(vecs)
    faiss.write_index(index, str(out_index))
    Path(out_manifest).write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return len(ids)


class FaissRetriever:
    """Live retriever over one or more (index, manifest) pairs sharing a corpus."""

    def __init__(self, embedder: Embedder, corpus: dict[str, str], k: int = 5):
        import faiss  # noqa: F401
        self.embedder = embedder
        self.corpus = corpus
        self.k = k
        self._indexes: list[tuple] = []   # (faiss_index, ids)

    def add(self, index_path: str | Path, manifest_path: str | Path) -> "FaissRetriever":
        import faiss
        idx = faiss.read_index(str(index_path))
        ids = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self._indexes.append((idx, ids))
        return self

    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        """Backwards-compatible: returns just the context texts."""
        _, texts = self.retrieve_with_ids(query, k)
        return texts

    def retrieve_with_ids(self, query: str,
                          k: int | None = None) -> tuple[list[str], list[str]]:
        """Return (ids, texts) of the top-k unique passages across all indexes."""
        k = k or self.k
        q = self.embedder.encode([query])
        hits: list[tuple[float, str]] = []
        for idx, ids in self._indexes:
            D, I = idx.search(q, min(k, len(ids)))
            for score, pos in zip(D[0], I[0]):
                if 0 <= pos < len(ids):
                    hits.append((float(score), ids[pos]))
        hits.sort(key=lambda x: x[0], reverse=True)
        out_ids, out_txt, seen = [], [], set()
        for _, doc_id in hits:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            txt = self.corpus.get(doc_id)
            if txt:
                out_ids.append(doc_id)
                out_txt.append(txt)
            if len(out_ids) >= k:
                break
        return out_ids, out_txt
