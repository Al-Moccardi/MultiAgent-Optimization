import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
"""
part4_dynamic_path/src/dynamic_lib.py
=====================================
Shared building blocks for Part 4 (dynamic, quality-aware agentic-path
selection). Imported by the numbered scripts 01..04; not run directly.

Part 4 keeps the Parts 1-3 pipeline structure (dispatcher -> specialists ->
synthesiser) and the fixed per-role allocation, adds NO second LLM, and uses
NO risk label and NO dispatcher confidence. It decides, per query, WHICH
specialist domains run -- making k(q) endogenous. The controller is built only
from:
  (1) retrieval geometry: query vs each domain's corpus passages, and
  (2) description geometry: query vs each agent's text description.
calibrated to a relevance probability and gated by a conformal coverage floor,
then a cost-aware submodular-coverage greedy that skips redundant domains.

Embedder: bge-m3 (production) via the repo's shared/faiss_code/embeddings.py,
else the dependency-free HashingEmbedder (offline PROXY, clearly logged).
Requires Python >= 3.10 (PEP 604 / builtin generics), matching the repo.
"""
import ast
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# repo's real embedders -- support both layouts:
#   replication repo: shared/faiss_code/embeddings.py
#   full codebase:    part1_allocation/scoring/embeddings.py
_sys.path.insert(0, str(_pl.Path(_P.SHARED) / "faiss_code"))
try:
    from embeddings import HashingEmbedder  # noqa: E402
except ModuleNotFoundError:
    _sys.path.insert(0, str(_pl.Path(_P.ROOT) / "part1_allocation" / "scoring"))
    from embeddings import HashingEmbedder  # noqa: E402


# ----------------------------------------------------------------------------- parsing
def parse_list(x):
    """Parse the repo's python-literal-string lists, e.g. "['A_x','A_y']"."""
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [t for t in x.split("|") if t]
    return x or []


def _read_text_robust(path) -> str:
    """Read a YAML/text file tolerant of Windows (cp1252) encodings."""
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


def load_queries(yaml_path) -> list[dict]:
    d = yaml.safe_load(_read_text_robust(yaml_path))
    items = d["queries"] if isinstance(d, dict) and "queries" in d else d
    # normalize the question field: some files use 'query' instead of 'text'
    for q in items:
        if "text" not in q and "query" in q:
            q["text"] = q["query"]
    return items


def get_embedder(prefer: str = "bge-m3"):
    """bge-m3 if importable+reachable, else HashingEmbedder PROXY."""
    if prefer and prefer.lower() != "hashing":
        try:
            from embeddings import STEmbedder
            emb = STEmbedder(model_name="BAAI/bge-m3")
            print(f"[embedder] bge-m3 (production), dim={emb.dim}")
            return emb, "bge-m3"
        except Exception as e:
            print(f"[embedder] bge-m3 unavailable ({type(e).__name__}); "
                  f"using HashingEmbedder PROXY.")
    emb = HashingEmbedder(1024)
    print(f"[embedder] HashingEmbedder PROXY, dim={emb.dim}")
    return emb, "hashing-proxy"


# ----------------------------------------------------------------------------- agents / corpus
def load_agents() -> tuple[list[str], dict[str, str]]:
    A = yaml.safe_load(Path(_P.AGENTS).read_text(encoding="utf-8"))["agents"]
    specialists = [a["id"] for a in A
                   if not a.get("is_dispatcher") and a["id"] != "A_synth"]
    desc = {a["id"]: (a.get("description") or a.get("name") or a["id"]).strip()
            for a in A}
    return specialists, desc


def domain_passage_texts(specialists: list[str]) -> dict[str, list[str]]:
    """Per-domain corpus profile: the TEXT of every gold authority attributed to
    a domain in the CALIBRATION set (calibration_with_gold.yaml). Calibration
    only -> no test leakage. The repo ships only corpus manifests (no passage
    texts), but the calibration gold passages carry their text, which is exactly
    what the retrieval-geometry signal needs."""
    queries = load_queries(_P.CALIBRATION)
    prof: dict[str, list[str]] = {s: [] for s in specialists}
    seen: dict[str, set] = {s: set() for s in specialists}
    for q in queries:
        gold = [g for g in parse_list(q.get("expected_agents")) if g in prof]
        if not gold:
            continue
        psgs = parse_list(q.get("gold_law_passages")) + parse_list(q.get("gold_case_passages"))
        for p in psgs:
            if isinstance(p, dict) and p.get("id") and p.get("text"):
                for d in gold:
                    if p["id"] not in seen[d]:
                        seen[d].add(p["id"]); prof[d].append(p["text"])
    return prof


# ----------------------------------------------------------------------------- signals
def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n).astype("float32") if n > 0 else v.astype("float32")


@dataclass
class DomainRelevance:
    embedder: object
    specialists: list[str]
    descriptions: dict[str, str]
    profile_texts: dict[str, list[str]]
    alpha_desc: float = 0.25
    # OPTIONAL frozen score cache: {query_text -> {domain -> {"ret_max","ret_mean",
    # "desc","fused"}}}. When a query is found here, its scores are REPLAYED verbatim
    # and the embedder is not called for it. This lets a machine WITHOUT bge-m3
    # reproduce the paper's numbers from a cache exported on the GPU (see
    # scripts/build_score_cache.py). Absent/empty -> pure live-embedder behaviour.
    score_cache: dict = field(default_factory=dict)
    _desc: dict[str, np.ndarray] = field(default_factory=dict)
    _mat: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Frozen domain-profile vectors (exported by scripts/build_score_cache.py
        # on the real embedder). The coverage-add stage of select_dynamic scores
        # candidates over these vectors, so freezing them makes the FULL
        # selection -- not just the gate -- reproducible without the live
        # embedder. MAMAP_REBUILD_VECTORS=1 forces live re-embedding (the
        # builder sets it).
        _vp = Path(__file__).resolve().parent.parent / "data" / "score_cache_vectors.npz"
        if os.environ.get("MAMAP_REBUILD_VECTORS") != "1" and _vp.exists():
            try:
                _z = np.load(_vp)
                if str(_z["embedder"]) == "bge-m3" and all(
                        f"desc__{s}" in _z.files and f"mat__{s}" in _z.files
                        for s in self.specialists):
                    for s in self.specialists:
                        self._desc[s] = _z[f"desc__{s}"]
                        self._mat[s] = _z[f"mat__{s}"]
                    print(f"[cache] loaded frozen domain-profile vectors "
                          f"(embedder=bge-m3) from {_vp.name}")
                    return
            except Exception as e:
                print(f"[cache] vector sidecar unreadable ({type(e).__name__}); "
                      f"falling back to live embedding.")
        for s in self.specialists:
            self._desc[s] = _l2(self.embedder.encode([self.descriptions.get(s, s)])[0])
            txt = self.profile_texts.get(s, [])
            self._mat[s] = (self.embedder.encode(txt) if txt
                            else np.zeros((0, self.embedder.dim), "float32"))

    def scores(self, query: str, candidates=None, topk: int = 5) -> dict[str, dict]:
        cands = candidates or list(self.specialists)
        # frozen-cache replay: if this exact query was cached, return it verbatim
        # (restricted to the requested candidates). No invented values are stored
        # here; the cache is only ever populated by build_score_cache.py on a real
        # embedder. A partial cache (some queries missing) falls through per-query.
        cached = self.score_cache.get(query)
        if cached is not None and all(d in cached for d in cands):
            return {d: dict(cached[d]) for d in cands}
        qv = _l2(self.embedder.encode([query])[0])
        out = {}
        for d in cands:
            P = self._mat.get(d)
            if P is not None and P.shape[0]:
                sims = P @ qv
                rmax = float(np.max(sims))
                k = min(topk, sims.shape[0])
                rmean = float(np.mean(np.sort(sims)[-k:]))
            else:
                rmax = rmean = 0.0
            dsc = float(self._desc[d] @ qv) if d in self._desc else 0.0
            fused = max(0.0, min(1.0, (1 - self.alpha_desc) * rmax + self.alpha_desc * dsc))
            out[d] = {"ret_max": rmax, "ret_mean": rmean, "desc": dsc, "fused": fused}
        return out

    def topk_vectors(self, query: str, domain: str, topk: int = 5) -> np.ndarray:
        P = self._mat.get(domain)
        if P is None or P.shape[0] == 0:
            return np.zeros((0, self.embedder.dim), "float32")
        qv = _l2(self.embedder.encode([query])[0])
        sims = P @ qv
        idx = np.argsort(sims)[-min(topk, P.shape[0]):]
        return P[idx]


# ----------------------------------------------------------------------------- calibration
class Isotonic:
    """PAV isotonic regression (monotone), no sklearn."""
    def __init__(self):
        self.x = np.array([0.0, 1.0]); self.y = np.array([0.0, 1.0])

    def fit(self, x, y):
        order = np.argsort(x)
        xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
        merged = []
        for j in range(len(ys)):
            merged.append([ys[j], 1.0, xs[j], xs[j]])
            while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
                v2, w2, lo2, hi2 = merged.pop()
                v1, w1, lo1, hi1 = merged.pop()
                nw = w1 + w2
                merged.append([(v1 * w1 + v2 * w2) / nw, nw, lo1, hi2])
        gx, gy = [], []
        for v, w_, lo, hi in merged:
            gx += [lo, hi]; gy += [v, v]
        self.x = np.array(gx); self.y = np.clip(np.array(gy), 0, 1)
        return self

    def predict(self, x):
        return np.interp(np.asarray(x, float), self.x, self.y)


@dataclass
class GateCalibration:
    """Maps a fused retrieval score -> calibrated relevance rho_hat (isotonic),
    and holds the inclusion threshold(s). Supports:
      - a single GLOBAL threshold (tau), or
      - PER-DOMAIN thresholds (tau_by_domain), one per specialist.
    Selected either by conformal risk control (alpha) or by maximizing F-beta of
    the kept set against gold on the calibration data (method='f2', beta=2 by
    default -> recall-favoring, appropriate when dropping a relevant domain is
    worse than running an extra one)."""
    iso: Isotonic
    tau: float                                  # global threshold (fallback)
    alpha: float
    n_cal: int
    method: str = "conformal"
    beta: float = 2.0
    tau_by_domain: dict[str, float] | None = None

    def rho_hat(self, fused: float) -> float:
        return float(self.iso.predict([fused])[0])

    def threshold_for(self, domain: str) -> float:
        if self.tau_by_domain and domain in self.tau_by_domain:
            return self.tau_by_domain[domain]
        return self.tau


def build_calibration_rows(rel: DomainRelevance) -> list[dict]:
    rows = []
    for q in load_queries(_P.CALIBRATION):
        gold = set(parse_list(q.get("expected_agents")))
        for d, sig in rel.scores(q["text"]).items():
            rows.append({"domain": d, "score": sig["fused"], "is_gold": int(d in gold)})
    return rows


def build_calibration_rows_from(rel: DomainRelevance, calib_path) -> list[dict]:
    """Same as build_calibration_rows but from an EXPLICIT calibration file
    (e.g. a larger expected_agents-only set used purely for threshold fitting,
    while the relevance PROFILES stay built from the gold-passage file)."""
    rows = []
    for q in load_queries(calib_path):
        gold = set(parse_list(q.get("expected_agents")))
        for d, sig in rel.scores(q["text"]).items():
            rows.append({"domain": d, "score": sig["fused"], "is_gold": int(d in gold)})
    return rows


def _fbeta_at(thresholds: np.ndarray, scores: np.ndarray, gold: np.ndarray,
              beta: float) -> tuple[float, float]:
    """Return (best_threshold, best_fbeta) over candidate thresholds for a binary
    keep decision rho_hat >= t. F-beta with beta>1 favors recall."""
    b2 = beta * beta
    best_t, best_f = float(thresholds[0]), -1.0
def _fbeta_at(thresholds: np.ndarray, scores: np.ndarray, gold: np.ndarray,
              beta: float, recall_floor: float = 0.0, tau_cap: float = 1.0) -> tuple[float, float]:
    """Return (best_threshold, best_fbeta) for keep := rho_hat >= t. beta>1 favors
    recall. `recall_floor` forbids thresholds whose calibration recall on positives
    falls below the floor (kills the degenerate 'set the bar so high nothing is
    kept' solution). `tau_cap` is an absolute ceiling on the chosen threshold."""
    b2 = beta * beta
    P = gold.sum()
    best_t, best_f = None, -1.0
    for t in thresholds:
        if t > tau_cap:
            continue
        keep = scores >= t
        tp = float((keep & (gold == 1)).sum())
        fp = float((keep & (gold == 0)).sum())
        rec = tp / P if P else 0.0
        if P and rec < recall_floor:          # recall too low -> disallow
            continue
        if tp == 0:
            f = 0.0
        else:
            prec = tp / (tp + fp)
            f = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
        if f > best_f + 1e-9:
            best_f, best_t = f, float(t)
    if best_t is None:                         # nothing satisfied the floor/cap
        best_t = float(min(thresholds.min(), tau_cap))
    return best_t, best_f


def calibrate_gate(rows: list[dict], alpha: float = 0.10,
                   method: str = "f2", beta: float = 2.0,
                   per_domain: bool = True,
                   min_per_domain_pos: int = 5,
                   recall_floor: float = 0.8,
                   tau_cap: float = 0.6,
                   shrink_k: float = 8.0) -> GateCalibration:
    """Fit the isotonic score->relevance map and choose inclusion threshold(s).

    method='conformal' : tau = conformal-risk-control quantile.
    method='f2'        : tau MAXIMIZES F-beta (beta=2 default, recall-favoring) of
                         the kept set vs gold on calibration, subject to:
                          - recall_floor: chosen tau must keep >= this fraction of
                            the domain's positives (forbids the degenerate 'bar so
                            high nothing passes' solution that caused over-abstention),
                          - tau_cap: hard ceiling on any threshold,
                          - min_per_domain_pos: domains with fewer positives use the
                            GLOBAL threshold instead of an overfit per-domain one,
                          - shrinkage: each per-domain tau is pulled toward the
                            global tau with weight shrink_k (empirical-Bayes style),
                            so thin-sample domains don't get extreme thresholds.
    """
    s = np.array([r["score"] for r in rows], float)
    g = np.array([r["is_gold"] for r in rows], float)
    dom = np.array([r["domain"] for r in rows])
    iso = Isotonic().fit(s, g)
    rho = iso.predict(s)

    if method == "conformal":
        rel = np.sort(rho[g == 1]); n = len(rel)
        if n == 0:
            return GateCalibration(iso, 0.0, alpha, 0, "conformal", beta, None)
        j = max(0, min(int(np.floor(alpha * (n + 1))), n - 1))
        return GateCalibration(iso, float(rel[j]), alpha, n, "conformal", beta, None)

    # ---- F-beta optimized threshold(s), robustified
    grid = np.unique(np.concatenate([[0.0], np.sort(rho), [1.0]]))
    g_tau, _ = _fbeta_at(grid, rho, g, beta, recall_floor, tau_cap)   # global
    tau_by_domain = None
    if per_domain:
        tau_by_domain = {}
        for d in sorted(set(dom)):
            m = dom == d
            n_pos = int(g[m].sum())
            if n_pos >= min_per_domain_pos and int((g[m] == 0).sum()) >= 1:
                raw, _ = _fbeta_at(np.unique(np.concatenate([[0.0], np.sort(rho[m]), [1.0]])),
                                   rho[m], g[m], beta, recall_floor, tau_cap)
                # shrink toward global: more positives -> trust the domain more
                td = (n_pos * raw + shrink_k * g_tau) / (n_pos + shrink_k)
                tau_by_domain[d] = float(min(td, tau_cap))
            else:
                tau_by_domain[d] = g_tau            # too few examples -> global
    return GateCalibration(iso, g_tau, alpha, int(g.sum()),
                           f"f{int(beta)}", beta, tau_by_domain)


# ----------------------------------------------------------------------------- measured quality
def specialist_quality_by_domain(spec_model: str) -> dict[str, float]:
    qt = pd.read_parquet(str(_P.QUALITY_TABLE))
    qt = qt[~qt.agent.isin(["A_dispatcher", "A_synth"])].dropna(subset=["quality"]).copy()
    qt["model"] = qt.config_id.str.split("__").str[0]
    pooled = qt.groupby("agent").quality.mean()
    sub = qt[qt.model == spec_model]
    bym = sub.groupby("agent").quality.mean() if len(sub) else pd.Series(dtype=float)
    return {a: float(bym[a]) if (a in bym.index and bym[a] == bym[a])
            else float(pooled.get(a, 0.6)) for a in qt.agent.unique()}


def synth_correctness_by_query(synth_model: str) -> tuple[dict[str, float], float]:
    """Measured synthesiser correctness per query for `synth_model`, from the
    quality table's q_correctness on the FULL activated set (A_synth rows).
    Returns ({query_id: correctness}, global_mean). This is the answer-quality
    metric for FULL activation; the DYNAMIC (pruned) version requires the synth
    re-run (04). query_id in the table may be 'GS001::A_synth' or 'GS001'."""
    qt = pd.read_parquet(str(_P.QUALITY_TABLE))
    s = qt[qt.agent == "A_synth"].copy()
    s["model"] = s.config_id.str.split("__").str[0]
    s = s[(s.model == synth_model) & s.q_correctness.notna()]
    by_q: dict[str, float] = {}
    for r in s.itertuples():
        qid = str(r.query_id).split("::")[0]
        by_q.setdefault(qid, []).append(float(r.q_correctness))
    by_q = {k: float(np.mean(v)) for k, v in by_q.items()}
    gm = float(np.mean(list(by_q.values()))) if by_q else float("nan")
    return by_q, gm


# ----------------------------------------------------------------------------- free (LLM-free) router
def free_router_candidates(rel: "DomainRelevance", query: str,
                           keep: float = 0.35, max_k: int = 6,
                           min_k: int = 1) -> list[str]:
    """A training-free, LLM-free DISPATCHER substitute: pre-filter the candidate
    domains from the SAME retrieval geometry the gate uses. Keeps domains whose
    fused score is within `keep` of the top domain's score (a soft margin),
    capped at max_k and floored at min_k. This is a real router (it can return a
    small, query-specific candidate set), but uses no extra LLM and no gold.

    The dynamic gate then prunes WITHIN this set; abstention is still possible
    because the conformal floor can reject all of them. Contrast with candidates
    = all 9 domains (no router) -- this is the realistic deployed setting where a
    cheap router proposes and the gate disposes."""
    sc = rel.scores(query)
    ranked = sorted(sc.items(), key=lambda kv: -kv[1]["fused"])
    top = ranked[0][1]["fused"] if ranked else 0.0
    cands = [d for d, s in ranked if s["fused"] >= top - keep][:max_k]
    if len(cands) < min_k:
        cands = [d for d, _ in ranked[:min_k]]
    return cands


# ----------------------------------------------------------------------------- cost model
@dataclass
class CostModel:
    L_disp: float; L_spec: float; L_synth: float
    E_disp: float; E_spec: float; E_synth: float
    mode: str = "concurrent"

    def latency(self, k: int) -> float:
        if k == 0:
            return self.L_disp
        spec = self.L_spec if self.mode == "concurrent" else k * self.L_spec
        return self.L_disp + spec + self.L_synth

    def energy(self, k: int) -> float:
        return self.E_disp if k == 0 else self.E_disp + k * self.E_spec + self.E_synth


def cost_model(disp_model="mistral-7b", spec_model="llama3.2-1b",
               synth_model="mistral-7b", mode="concurrent") -> CostModel:
    """Measured cost at the fixed allocation, from PERF_TABLE. Picks the cheapest
    feasible config (>= role's min context) for each role's model."""
    perf = pd.read_parquet(str(_P.PERF_TABLE))
    perf["model"] = perf.config_id.str.split("__").str[0]
    perf["lat_disp"] = perf.ttft_s + _P.TOK_ROUTING / perf.throughput_tok_s
    perf["lat_gen"] = perf.ttft_s + _P.TOK_GENERATION / perf.throughput_tok_s
    perf = perf[perf.peak_mem_gb <= _P.MEM_BUDGET_GB]

    def pick(model):
        sub = perf[perf.model == model].sort_values("peak_mem_gb")
        if not len(sub):
            sub = perf.sort_values("peak_mem_gb")
        return sub.iloc[0]
    d, s, y = pick(disp_model), pick(spec_model), pick(synth_model)
    epb = lambda r, n: float(r.energy_j_per_tok) * n
    return CostModel(
        L_disp=float(d.lat_disp), L_spec=float(s.lat_gen), L_synth=float(y.lat_gen),
        E_disp=epb(d, _P.TOK_ROUTING), E_spec=epb(s, _P.TOK_GENERATION),
        E_synth=epb(y, _P.TOK_GENERATION), mode=mode)


def budget_from_entropy(scores: dict[str, dict], cost_one: float,
                        k_min: int = 1, k_max: int = 6) -> float:
    """LLM-free, risk-free per-query budget: diffuse retrieval -> harder -> larger
    budget. Hardness = normalized entropy of softmax over candidate fused scores."""
    v = np.array([s["fused"] for s in scores.values()], float)
    if v.size == 0:
        return cost_one * k_min
    p = np.exp(v - v.max()); p = p / p.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12)
    return float(cost_one * (k_min + ent * (k_max - k_min)))


# ----------------------------------------------------------------------------- selector
@dataclass
class Candidate:
    domain: str; rho_hat: float; quality: float; cost: float; vectors: np.ndarray
    thr: float = 0.5            # this domain's inclusion threshold (from GateCalibration)


@dataclass
class Selection:
    domains: list[str]; k: int; cost: float; covered: float
    reason: dict[str, str] = field(default_factory=dict)


def _coverage_gain(sel_vecs: list[np.ndarray], cand: np.ndarray, w: float) -> float:
    """Facility-location marginal coverage gain (monotone submodular; greedy is
    1-1/e). Weighted by relevance*quality so coverage credits useful content."""
    if cand.shape[0] == 0:
        return 0.0
    if not sel_vecs:
        return float(w * cand.shape[0])
    S = np.vstack(sel_vecs)
    best = np.clip((cand @ S.T).max(axis=1), 0, 1)
    return float(w * float(np.sum(np.clip(1.0 - best, 0, 1))))


def _total_coverage(cs: list[Candidate]) -> float:
    vecs, tot = [], 0.0
    for c in cs:
        tot += _coverage_gain(vecs, c.vectors, max(1e-6, c.rho_hat * c.quality))
        if c.vectors.shape[0]:
            vecs.append(c.vectors)
    return float(tot)


def select_dynamic(cands: list[Candidate], tau: float | None = None, budget=None,
                   min_gain: float = 0.15) -> Selection:
    """Floor (rho_hat >= the domain's own threshold, never dropped) + cost-aware
    submodular coverage adds. If `tau` is given it overrides every domain's
    threshold (global mode); otherwise each candidate uses its per-domain `thr`."""
    def thr(c):
        return tau if tau is not None else c.thr
    reason = {}
    must = [c for c in cands if c.rho_hat >= thr(c)]
    rest = [c for c in cands if c.rho_hat < thr(c)]
    for c in must:
        reason[c.domain] = "floor"
    sel = list(must); vecs = [c.vectors for c in sel if c.vectors.shape[0]]
    cost = sum(c.cost for c in sel)
    pool = list(rest)
    while pool:
        best, bratio, bgain = None, 0.0, 0.0
        for c in pool:
            g = _coverage_gain(vecs, c.vectors, max(1e-6, c.rho_hat * c.quality))
            r = g / max(c.cost, 1e-6)
            if r > bratio:
                best, bratio, bgain = c, r, g
        if best is None or bgain < min_gain:
            break
        if budget is not None and cost + best.cost > budget:
            break
        sel.append(best); cost += best.cost; reason[best.domain] = "coverage_add"
        if best.vectors.shape[0]:
            vecs.append(best.vectors)
        pool.remove(best)
    return Selection([c.domain for c in sel], len(sel), cost, _total_coverage(sel), reason)


def select_threshold(cands, tau=None):
    def thr(c):
        return tau if tau is not None else c.thr
    keep = [c for c in cands if c.rho_hat >= thr(c)]
    return Selection([c.domain for c in keep], len(keep), sum(c.cost for c in keep),
                     _total_coverage(keep), {c.domain: "threshold" for c in keep})


def select_topk(cands, k):
    keep = sorted(cands, key=lambda c: -c.rho_hat)[:k]
    return Selection([c.domain for c in keep], len(keep), sum(c.cost for c in keep),
                     _total_coverage(keep), {c.domain: "topk" for c in keep})


def select_full(cands):
    return Selection([c.domain for c in cands], len(cands), sum(c.cost for c in cands),
                     _total_coverage(cands), {c.domain: "full" for c in cands})


# ----------------------------------------------------------------------------- allocation presets
# Three fixed allocations the dynamic selector runs ON TOP OF, grounded in the
# Part-2 (additive) and Part-3 (gated) frontier outputs in shared/pareto/.
# Each preset = (dispatcher_model, specialist_model, synthesiser_model). The
# specialist is Llama-3.2-1B in the optimised presets because it is the
# quality-aware optimum at essentially every budget on both frontiers.
ALLOC_PRESETS = {
    # Part-2 quality-aware optimum (the headline "mixed" optimum)
    "qa_optimum":    ("mistral-7b",   "llama3.2-1b", "mistral-7b"),
    # Part-3 gated/bilinear optimum: small, high-recall router
    "gated_optimum": ("smollm2-360m", "llama3.2-1b", "mistral-7b"),
    # single-family contrast (Part-2 §single-family ablation): all-Qwen
    "qwen":          ("qwen2.5-3b",   "qwen2.5-3b",  "qwen2.5-3b"),
}
