"""
part1_allocation/scoring/embeddings.py
======================================
Text embedders for (a) FAISS retrieval and (b) RAGAS-style answer_relevancy /
semantic correctness.

Default (recommended): **BAAI/bge-m3** -- multilingual (incl. Italian), 1024-dim,
which matches the project's existing FAISS dimension. Needs `sentence-transformers`
and a one-time model download.

Offline fallback: **HashingEmbedder** -- dependency-free, deterministic, 1024-dim.
Lets the entire pipeline (FAISS build + retrieval + relevancy) run with NO model
download (used when sentence-transformers / the HF hub is unreachable). It is a
bag-of-words hashing embedder: fine to validate the plumbing, but use bge-m3 for
real semantic quality.

All embedders return L2-normalized float32 vectors so an inner-product FAISS index
(IndexFlatIP) computes cosine similarity directly.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np


class Embedder:
    dim: int

    def encode(self, texts: list[str]) -> "np.ndarray":  # (n, dim) float32, normalized
        raise NotImplementedError

    def encode_one(self, text: str) -> "np.ndarray":
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    """Offline, deterministic hashing embedder (unigrams + bigrams)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> "np.ndarray":
        v = np.zeros(self.dim, dtype=np.float32)
        toks = re.findall(r"\w+", (text or "").lower())
        grams = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:12], 16)
            v[h % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def encode(self, texts: list[str]) -> "np.ndarray":
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts]).astype("float32")


class STEmbedder(Embedder):
    """sentence-transformers backend (e.g. BAAI/bge-m3). device: None=auto, 'cpu', 'cuda'."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name, device=device)
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: list[str]) -> "np.ndarray":
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        v = self._m.encode(list(texts), normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(v, dtype="float32")


_ALIASES = {
    "bge-m3": "BAAI/bge-m3",
    "e5": "intfloat/multilingual-e5-large",
    "e5-large": "intfloat/multilingual-e5-large",
}


def make_embedder(name: str = "bge-m3", dim: int = 1024,
                  device: str | None = None) -> Embedder:
    """Resolve an embedder by short name. `device`: None=auto, 'cpu', 'cuda' (use
    'cpu' to keep VRAM free for the judge/candidate models). Falls back to the
    offline HashingEmbedder if the model can't be loaded."""
    if name in ("hash", "offline", "none", ""):
        return HashingEmbedder(dim=dim)
    model = _ALIASES.get(name, name)
    try:
        return STEmbedder(model, device=device)
    except Exception as e:
        print(f"[embed] '{model}' unavailable ({type(e).__name__}: {e}); "
              f"falling back to offline HashingEmbedder({dim}d).")
        return HashingEmbedder(dim=dim)


def cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    """Cosine of two (already L2-normalized) vectors -> inner product."""
    return float(np.dot(a, b))
