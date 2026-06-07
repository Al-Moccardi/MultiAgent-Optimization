"""
part4_dynamic_path/src/select.py
================================
The dynamic agentic-path SELECTOR. Given a query's candidate domains and their
LLM-free signals, choose the subset S(q) of specialists to actually run.

Objective (legal-appropriate, the orientation chosen for this pipeline):

        minimize   cost(S)
        subject to a CONFORMAL COVERAGE floor on gold-relevant domains,
                   i.e. keep every domain whose calibrated relevance
                   rho_hat_d >= tau (tau set by conformal risk control so the
                   expected miss-rate of relevant domains <= alpha),
        then, within the remaining budget, ADD domains by submodular COVERAGE
        so the selected set covers the query's information need with little
        redundancy (relevant-but-redundant domains add ~nothing and are skipped).

This yields an ENDOGENOUS k(q) = |S(q)|: the number of activated specialists is
decided per query, not swept as a parameter (the dynamic-k analogue of Parts
1-3's fixed-k frontier).

Selectors implemented (for the head-to-head in the paper):
  - "dynamic"   : conformal floor + cost-aware submodular coverage  (primary)
  - "threshold" : keep all candidates with rho_hat >= tau           (ablation)
  - "topk"      : keep the k highest-rho_hat candidates             (baseline)
  - "full"      : keep all candidates (no pruning)                  (baseline)

NO LLM is used anywhere here. Coverage uses the embedding geometry already
computed by signals.DomainRelevance; cost uses the measured perf table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class Candidate:
    domain: str
    rho_hat: float                  # calibrated relevance in [0,1]
    quality: float                  # measured Q_spec(M_s, domain) in [0,1]
    cost: float                     # marginal cost to run this specialist (s or J)
    vectors: np.ndarray             # (n_topk, d) retrieved-passage embeddings


@dataclass
class Selection:
    domains: list[str]
    k: int
    cost: float
    covered: float                  # achieved coverage value F(S)
    reason: dict[str, str] = field(default_factory=dict)  # domain -> why kept


# --------------------------------------------------------------------------- coverage
def _coverage_gain(selected_vecs: list[np.ndarray], cand_vecs: np.ndarray,
                   weight: float) -> float:
    """Facility-location-style marginal coverage gain of adding `cand_vecs`.

    Coverage of a passage set is the sum over the query's retrieved 'facets'
    (here: the candidate's own top passages act as facets) of the best
    similarity to anything already selected. Monotone & submodular, so greedy
    has the (1 - 1/e) guarantee (Nemhauser-Wolsey). Weighted by the domain's
    relevance*quality so coverage credits useful content, not just any content.
    """
    if cand_vecs.shape[0] == 0:
        return 0.0
    if not selected_vecs:
        # full self-coverage of the candidate's facets
        return float(weight * cand_vecs.shape[0])
    S = np.vstack(selected_vecs)                       # (m, d)
    # for each candidate facet, how well is it ALREADY covered by S?
    sims = cand_vecs @ S.T                              # (n, m)
    best_existing = np.clip(sims.max(axis=1), 0.0, 1.0)  # (n,)
    novelty = np.clip(1.0 - best_existing, 0.0, 1.0)     # uncovered mass
    return float(weight * float(np.sum(novelty)))


# --------------------------------------------------------------------------- selectors
def select_dynamic(cands: list[Candidate], tau: float,
                   budget: Optional[float] = None,
                   min_gain: float = 0.15) -> Selection:
    """Conformal floor + cost-aware submodular coverage.

    1. MUST-KEEP: every candidate with rho_hat >= tau (the conformal guarantee).
    2. If a budget is given and the must-keep set already exceeds it, we do NOT
       drop below the floor (safety first) -- we keep the floor and flag it.
    3. With remaining budget, greedily add the candidate with the best
       coverage-gain-per-cost while marginal gain >= min_gain.
    """
    reason: dict[str, str] = {}
    must = [c for c in cands if c.rho_hat >= tau]
    rest = [c for c in cands if c.rho_hat < tau]
    for c in must:
        reason[c.domain] = "conformal_floor"

    selected = list(must)
    sel_vecs = [c.vectors for c in selected if c.vectors.shape[0] > 0]
    cost = sum(c.cost for c in selected)

    # cost-aware submodular additions from the sub-threshold remainder
    pool = list(rest)
    while pool:
        best, best_ratio, best_gain = None, 0.0, 0.0
        for c in pool:
            w = max(1e-6, c.rho_hat * c.quality)
            gain = _coverage_gain(sel_vecs, c.vectors, w)
            ratio = gain / max(c.cost, 1e-6)
            if ratio > best_ratio:
                best, best_ratio, best_gain = c, ratio, gain
        if best is None or best_gain < min_gain:
            break
        if budget is not None and (cost + best.cost) > budget:
            break
        selected.append(best)
        sel_vecs.append(best.vectors) if best.vectors.shape[0] > 0 else None
        cost += best.cost
        reason[best.domain] = "coverage_add"
        pool.remove(best)

    covered = _total_coverage(selected)
    doms = [c.domain for c in selected]
    return Selection(domains=doms, k=len(doms), cost=cost,
                     covered=covered, reason=reason)


def select_threshold(cands: list[Candidate], tau: float) -> Selection:
    keep = [c for c in cands if c.rho_hat >= tau]
    return Selection([c.domain for c in keep], len(keep),
                     sum(c.cost for c in keep), _total_coverage(keep),
                     {c.domain: "threshold" for c in keep})


def select_topk(cands: list[Candidate], k: int) -> Selection:
    keep = sorted(cands, key=lambda c: -c.rho_hat)[:k]
    return Selection([c.domain for c in keep], len(keep),
                     sum(c.cost for c in keep), _total_coverage(keep),
                     {c.domain: "topk" for c in keep})


def select_full(cands: list[Candidate]) -> Selection:
    return Selection([c.domain for c in cands], len(cands),
                     sum(c.cost for c in cands), _total_coverage(cands),
                     {c.domain: "full" for c in cands})


def _total_coverage(cands: list[Candidate]) -> float:
    vecs: list[np.ndarray] = []
    total = 0.0
    for c in cands:
        w = max(1e-6, c.rho_hat * c.quality)
        total += _coverage_gain(vecs, c.vectors, w)
        if c.vectors.shape[0] > 0:
            vecs.append(c.vectors)
    return float(total)
