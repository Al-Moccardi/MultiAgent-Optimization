"""
part4_dynamic_path/src/signals.py
=================================
LLM-FREE per-domain relevance signals for dynamic agentic-path selection.

The dynamic layer decides, per query, WHICH specialist domains actually run --
it never changes which model serves a role (that is Parts 1-3), never adds an
agent, and never calls a second LLM. The only runtime signals are:

  (1) RETRIEVAL geometry: similarity of the query to each domain's corpus
      passages (the same FAISS retrieval the specialist would do anyway), and
  (2) DESCRIPTION geometry: similarity of the query to each agent's text
      description (a cheap, always-available domain prototype).

Both use the project's existing `Embedder` (bge-m3 in production; the
dependency-free HashingEmbedder offline). No dispatcher confidence is used
(the deployed dispatcher does not expose per-domain scores) and no risk label
is used (not available).

A per-domain corpus profile is built ONCE, offline, by attributing each gold
supporting authority in the calibration set to the domain(s) that cited it
(`calib_clean_with_gold_text.yaml`). At runtime a domain's retrieval score is
the max / mean / softmax-mass of query-passage cosine similarities over that
domain's profile passages, restricted to passages that are actually in the
live corpus index.
"""
from __future__ import annotations

import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


# ----------------------------------------------------------------------------- helpers
def _parse(x):
    """calib fields are python-literal strings like "['A_x', 'A_y']"."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def load_corpus(jsonl_path: str | Path) -> dict[str, str]:
    corpus: dict[str, str] = {}
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        corpus[r["id"]] = r["text"]
    return corpus


def load_agents(agents_yaml: str | Path) -> tuple[list[str], dict[str, str]]:
    """Return (specialist_ids, {agent_id: description}) for all agents."""
    A = yaml.safe_load(Path(agents_yaml).read_text(encoding="utf-8"))["agents"]
    specialists = [a["id"] for a in A
                   if not a.get("is_dispatcher") and a["id"] != "A_synth"]
    desc = {a["id"]: (a.get("description") or a.get("name") or a["id"]).strip()
            for a in A}
    return specialists, desc


def domain_passage_profile(calib_yaml: str | Path,
                           specialists: list[str]) -> dict[str, list[str]]:
    """Attribute each calibration gold passage id to the domain(s) that cited it.

    Returns {domain_id: [passage_id, ...]} -- the per-domain corpus profile used
    to score query<->domain retrieval similarity. Pure bookkeeping over gold
    annotations; no model and no test-set leakage (calibration set only).
    """
    queries = yaml.safe_load(Path(calib_yaml).read_text(encoding="utf-8"))["queries"]
    prof: dict[str, set[str]] = {s: set() for s in specialists}
    for q in queries:
        gold = set(_parse(q.get("expected_agents")) or [])
        gold = {g for g in gold if g in prof}
        if not gold:
            continue
        psgs = (_parse(q.get("gold_law_passages")) or []) + \
               (_parse(q.get("gold_case_passages")) or [])
        for p in psgs:
            if isinstance(p, dict) and p.get("id"):
                for d in gold:
                    prof[d].add(p["id"])
    return {d: sorted(v) for d, v in prof.items()}


# ----------------------------------------------------------------------------- core
@dataclass
class DomainRelevance:
    """Computes LLM-free per-domain relevance scores for a query.

    Parameters
    ----------
    embedder : Embedder
        Project embedder (bge-m3 in production, HashingEmbedder offline).
    corpus : dict[str, str]
        id -> passage text (the live corpus the retriever indexes).
    specialists : list[str]
    descriptions : dict[str, str]
        agent_id -> description text.
    domain_profile : dict[str, list[str]]
        domain_id -> passage ids (from `domain_passage_profile`).
    alpha_desc : float
        Weight on the description-similarity signal when fused with retrieval
        (0 => retrieval only; 1 => description only). Default 0.25.
    """
    embedder: object
    corpus: dict[str, str]
    specialists: list[str]
    descriptions: dict[str, str]
    domain_profile: dict[str, list[str]]
    alpha_desc: float = 0.25

    # filled in __post_init__
    _desc_vec: dict[str, np.ndarray] = field(default_factory=dict)
    _profile_mat: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # description prototypes
        for s in self.specialists:
            v = self.embedder.encode([self.descriptions.get(s, s)])[0]
            self._desc_vec[s] = _l2(v)
        # per-domain passage matrices (only passages present in the live corpus)
        for s in self.specialists:
            texts = [self.corpus[i] for i in self.domain_profile.get(s, [])
                     if i in self.corpus]
            if texts:
                self._profile_mat[s] = self.embedder.encode(texts)  # (n, d), normalized
            else:
                self._profile_mat[s] = np.zeros((0, self.embedder.dim), dtype="float32")

    # -------- per-query scoring
    def scores(self, query: str, candidates: list[str] | None = None,
               topk: int = 5) -> dict[str, dict[str, float]]:
        """Return per-domain signal dict for the given query.

        For each domain d in `candidates` (default: all specialists):
          ret_max  : max cosine(query, passage) over d's profile
          ret_mean : mean of top-`topk` cosines
          desc     : cosine(query, d's description prototype)
          fused    : (1-alpha)*ret_max + alpha*desc      [in [0,1] after clip]
        """
        cands = candidates or list(self.specialists)
        qv = _l2(self.embedder.encode([query])[0])
        out: dict[str, dict[str, float]] = {}
        for d in cands:
            P = self._profile_mat.get(d)
            if P is not None and P.shape[0] > 0:
                sims = P @ qv
                ret_max = float(np.max(sims))
                k = min(topk, sims.shape[0])
                ret_mean = float(np.mean(np.sort(sims)[-k:]))
            else:
                ret_max = ret_mean = 0.0
            dsc = float(self._desc_vec[d] @ qv) if d in self._desc_vec else 0.0
            fused = (1.0 - self.alpha_desc) * ret_max + self.alpha_desc * dsc
            out[d] = {"ret_max": ret_max, "ret_mean": ret_mean,
                      "desc": dsc, "fused": max(0.0, min(1.0, fused))}
        return out

    def retrieved_vectors(self, query: str, domain: str,
                          topk: int = 5) -> np.ndarray:
        """Top-`topk` profile-passage embeddings for (query, domain), for the
        submodular COVERAGE objective (so redundant domains add little)."""
        P = self._profile_mat.get(domain)
        if P is None or P.shape[0] == 0:
            return np.zeros((0, self.embedder.dim), dtype="float32")
        qv = _l2(self.embedder.encode([query])[0])
        sims = P @ qv
        idx = np.argsort(sims)[-min(topk, P.shape[0]):]
        return P[idx]


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype("float32") if n > 0 else v.astype("float32")
