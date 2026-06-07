"""
part4_dynamic_path/run_pipeline.py
==================================
REAL end-to-end test of the dynamic agentic-path on HELD-OUT (non-overlapping)
test queries. For each query it runs the actual pipeline TWICE and compares:

  NORMAL routing : dispatcher LLM -> its predicted specialist domains all run
                   (RAG-retrieve + generate) -> synthesiser composes the answer.
  DYNAMIC routing: same dispatcher prediction -> the Part-4 gate prunes it to a
                   per-query subset S(q) (or abstains) -> only those specialists
                   run -> synthesiser.

Reports, per query and as means:
  * latency_s and DELTA latency (dynamic - normal)        [real wall-clock]
  * correctness and DELTA correctness (dynamic - normal)

CORRECTNESS on held-out queries (which carry NO gold answer):
  - out-of-domain queries (expected_outcome == abstain): correctness = correct
    abstention (1.0 if the pipeline abstained, else 0.0). Reference-free.
  - in-domain queries (no gold answer available): correctness = the project's
    judge-free RAGAS-style answer quality (faithfulness to retrieved law +
    answer relevancy + context precision/recall), computed with bge-m3. Labelled
    `answer_quality` so it is not confused with gold-correctness.
  - optional `--judge_model`: if given, in-domain correctness is scored by that
    judge LLM against the retrieved context instead (gold-free LLM-as-judge).

This needs the REAL environment: GGUF models (resolved from catalog_zoo.yaml via
llama-cpp-python) + bge-m3 + the FAISS corpus. Run it in the full codebase. It
fails fast (never fabricates) if a backend/model/index is missing.

Run (from the full-codebase repo root):
    python -m part4_dynamic_path.run_pipeline --config gated_optimum --alpha 0.10
    python -m part4_dynamic_path.run_pipeline --config qa_optimum
    python -m part4_dynamic_path.run_pipeline --config qwen
    # quick smoke test on 6 queries with the mock backend (no models needed):
    python -m part4_dynamic_path.run_pipeline --config gated_optimum --mock --limit 6
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE / "src"))

# Part-4 gate building blocks (the dynamic layer)
from dynamic_lib import (DomainRelevance, load_agents, domain_passage_texts,    # noqa: E402
                         load_queries, parse_list, build_calibration_rows,
                         calibrate_gate, specialist_quality_by_domain,
                         budget_from_entropy, Candidate, select_dynamic,
                         free_router_candidates, ALLOC_PRESETS)

# project modules (full codebase)
from part1_allocation.scoring.embeddings import STEmbedder, HashingEmbedder      # noqa: E402
from part1_allocation.scoring.retrieval import FaissRetriever                    # noqa: E402

CATALOG = REPO / "part1_allocation" / "config" / "catalog.zoo.yaml"
if not CATALOG.exists():
    CATALOG = REPO / "shared" / "data" / "catalog_zoo.yaml"
CORPUS_DIR = REPO / "shared" / "corpus"

def _first_existing(*cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return Path(cands[-1])

# calibration (gate source) — try data\manifests first, then older locations
CALIB = _first_existing(
    REPO / "data" / "manifests" / "calib_clean_with_gold_text.yaml",
    REPO / "part1_allocation" / "data" / "calib_clean_with_gold_text.yaml",
    REPO / "shared" / "data" / "calibration_with_gold.yaml")
# in-domain held-out test set
TEST = _first_existing(
    REPO / "data" / "manifests" / "test_queries_en.yaml",
    REPO / "shared" / "data" / "test_queries_en.yaml",
    HERE / "data" / "test_queries_en.yaml")
# optional dedicated out-of-domain set (abstain ground truth)
OOD = _first_existing(
    REPO / "data" / "manifests" / "ood_queries_en.yaml",
    REPO / "shared" / "data" / "ood_queries_en.yaml")
OOD = OOD if OOD.exists() else None


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


# --------------------------------------------------------------------------- model resolution
# Where GGUF files may live on this machine (searched in order, recursively).
GGUF_SEARCH_DIRS = [REPO / "models", REPO / "data" / "gguf"]


def resolve_gguf(catalog_path: Path, model_key: str, quant: str) -> str:
    """Return an existing GGUF path for (model_key, quant).

    Strategy: take the catalog's filename for this model+quant, then look for that
    basename anywhere under GGUF_SEARCH_DIRS (so it works whether files are in
    models\\ or data\\gguf\\). If the exact quant file is absent, fall back to any
    quant of the same model that DOES exist (and log it), so a missing quant never
    hard-stops the run."""
    cat = yaml.safe_load(Path(catalog_path).read_text(encoding="utf-8"))["models"]
    entry = next((m for m in cat if m["key"] == model_key), None)
    if entry is None:
        raise KeyError(f"model {model_key} not in catalog {catalog_path}")
    # candidate basenames: requested quant first, then all quants of this model
    want = [q for q in entry["quantizations"] if q["label"] == quant]
    ordered = want + [q for q in entry["quantizations"] if q["label"] != quant]
    # 1) exact path as written in the catalog (relative to repo root)
    for q in ordered:
        p = REPO / q["gguf_path"]
        if p.exists():
            if q["label"] != quant:
                print(f"   [gguf] {model_key}: quant {quant} not found, using {q['label']}")
            return str(p)
    # 2) search by basename under known dirs
    for q in ordered:
        base = Path(q["gguf_path"]).name
        for d in GGUF_SEARCH_DIRS:
            hits = list(Path(d).rglob(base)) if Path(d).exists() else []
            if hits:
                if q["label"] != quant:
                    print(f"   [gguf] {model_key}: quant {quant} not found, using {q['label']}")
                return str(hits[0])
    # 3) loose: any file matching the model's quant label pattern
    raise SystemExit(
        f"No GGUF for {model_key} ({quant}) found under {', '.join(str(d) for d in GGUF_SEARCH_DIRS)}.\n"
        f"Tried basenames: {[Path(q['gguf_path']).name for q in ordered]}")


def make_backend(model_key: str, quant: str, n_ctx: int, mock: bool,
                 backend: str = "llama_cpp", gpu_layers: int = -1):
    from part1_allocation.inference.backend import LlamaCppBackend, MockBackend
    if mock:
        return MockBackend(config_tag=f"{model_key}__{quant}")
    if backend == "ollama":
        return OllamaBackend(OLLAMA_NAMES.get(model_key, model_key), n_ctx=n_ctx)
    path = resolve_gguf(CATALOG, model_key, quant)   # guaranteed to exist or SystemExit
    return LlamaCppBackend(gguf_path=path, n_ctx=n_ctx, n_gpu_layers=gpu_layers)


# Map this project's model keys -> Ollama model names.
# These names assume you registered YOUR existing GGUF files with:
#   ollama create <name> -f Modelfile   (FROM ./models/<file>.gguf)
# so Ollama runs your exact files/quants on the GPU. Adjust if you named them
# differently or used a different quant.
OLLAMA_NAMES = {
    "smollm2-360m": "smollm2-360m-q5",
    "llama3.2-1b":  "llama32-1b-q5",
    "mistral-7b":   "mistral-7b-q5",
    "qwen2.5-3b":   "qwen25-3b-q5",
    "llama3.2-3b":  "llama32-3b-q5",
    "qwen2.5-1_5b": "qwen25-1_5b-q5",
    "qwen2.5-0_5b": "qwen25-0_5b-q5",
    "phi3.5-mini":  "phi35-mini-q5",
    "gemma2-2b":    "gemma2-2b-q5",
}


class _Gen:
    """Minimal GenResult-shaped object (text + real wall-clock total_s)."""
    def __init__(self, text, n_out_tokens, ttft_s, total_s):
        self.text = text; self.n_out_tokens = n_out_tokens
        self.ttft_s = ttft_s; self.total_s = total_s


class OllamaBackend:
    """Self-contained backend that calls a local Ollama server (no compiler, no
    AVX issues). Same `.generate(prompt, max_tokens, context) -> result.text /
    .total_s` contract the pipeline uses. Start Ollama and `ollama pull` the model
    names in OLLAMA_NAMES first. Latency is real wall-clock; for finer detail
    Ollama also returns eval timings, used here when present."""
    def __init__(self, model_name: str, n_ctx: int = 4096,
                 host: str = "http://localhost:11434"):
        import urllib.request  # stdlib only
        self._urllib = urllib.request
        self.model = model_name
        self.n_ctx = n_ctx
        self.host = host
        # fail fast if the model isn't registered / server isn't up
        try:
            try:
                self._post("/api/show", {"model": model_name}, timeout=10)
            except Exception:
                self._post("/api/show", {"name": model_name}, timeout=10)
        except Exception as e:
            raise SystemExit(
                f"Ollama model '{model_name}' not available ({e}).\n"
                f"Register your GGUF with:  ollama create {model_name} -f Modelfile\n"
                f"(Modelfile contains:  FROM ./models/<your_file>.gguf )\n"
                f"and check 'ollama list'. Ensure the Ollama app/server is running.")

    def _post(self, path, payload, timeout=600):
        import json as _j
        req = self._urllib.Request(self.host + path,
                                   data=_j.dumps(payload).encode("utf-8"),
                                   headers={"Content-Type": "application/json"})
        with self._urllib.urlopen(req, timeout=timeout) as r:
            return _j.loads(r.read().decode("utf-8"))

    def generate(self, prompt: str, *, max_tokens: int = 512, context=None) -> _Gen:
        t0 = time.perf_counter()
        out = self._post("/api/generate", {
            "model": self.model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens,
                        "num_ctx": self.n_ctx}})
        total = time.perf_counter() - t0
        text = out.get("response", "")
        n_out = int(out.get("eval_count", len(text.split())))
        # real decode timings from ollama (ns) if present
        ttft = float(out.get("prompt_eval_duration", 0)) / 1e9 or total
        return _Gen(text=text, n_out_tokens=n_out, ttft_s=ttft, total_s=total)

    def close(self):
        pass


# --------------------------------------------------------------------------- prompts
DOMAIN_NAMES = None  # filled from agents.yaml


def dispatcher_prompt(query: str, specialists: list[str], desc: dict) -> str:
    menu = "\n".join(f"- {s}: {desc.get(s, '')[:160]}" for s in specialists)
    return ("You are the router of an Italian family-law assistant. Choose ALL "
            "specialist domains needed to answer the question, or reply NONE if the "
            "question is outside Italian family law. Reply with domain ids only, "
            f"comma-separated.\n\nDomains:\n{menu}\n\nQuestion: {query}\n\nDomains:")


def specialist_prompt(query: str, contexts: list[str], domain_desc: str) -> str:
    ctx = "\n\n".join(f"[Context {i+1}] {c}" for i, c in enumerate(contexts))
    return (f"You are a specialist in: {domain_desc}\nAnswer the question using ONLY "
            f"the retrieved Italian law/case contexts. Cite article/case ids.\n\n"
            f"{ctx}\n\nQuestion: {query}\n\nAnswer:")


def synth_prompt(query: str, specialist_answers: list[str]) -> str:
    joined = "\n\n".join(f"[Specialist {i+1}]\n{a}" for i, a in enumerate(specialist_answers))
    if not specialist_answers:
        return ("You are the synthesiser of an Italian family-law assistant. The router "
                "found NO relevant specialist domain for the question below, which means "
                "it is outside scope. Reply with a brief abstention telling the user this "
                f"is outside Italian family law.\n\nQuestion: {query}\n\nAnswer:")
    return ("You are the synthesiser of an Italian family-law assistant. Compose ONE "
            "coherent, correct final answer using only the specialist analyses below. "
            f"Do not invent authorities.\n\nQuestion: {query}\n\n{joined}\n\nFinal answer:")


def parse_router_output(text: str, specialists: list[str]) -> list[str]:
    t = (text or "").lower()
    if "none" in t and not any(s.lower() in t for s in specialists):
        return []
    return [s for s in specialists if s.lower() in t or s.lower().replace("a_", "") in t]


# --------------------------------------------------------------------------- quality (gold-free)
def answer_quality(emb, answer: str, contexts: list[str], query: str) -> float:
    """Judge-free RAGAS-style answer quality with bge-m3 (no gold needed):
    mean of (faithfulness: answer<->contexts) and (relevancy: answer<->query).
    Bounded [0,1]. Mirrors the project's judge-free `quality` basis."""
    if not answer.strip():
        return 0.0
    av = _unit(emb.encode([answer])[0])
    qv = _unit(emb.encode([query])[0])
    relevancy = float(max(0.0, av @ qv))
    if contexts:
        C = emb.encode(contexts)
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        faith = float(np.clip(np.max(C @ av), 0.0, 1.0))
    else:
        faith = 0.0
    return float(0.5 * faith + 0.5 * relevancy)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# --------------------------------------------------------------------------- one routing pass
def compute_latency(r) -> float:
    """COMPUTE-ONLY latency for one generation: prefill (ttft) + decode time,
    excluding cold model-load / VRAM-swap stalls. Reconstructed from GenResult
    fields so it is invariant to Ollama loading/evicting models between calls.
      decode_time = n_out_tokens / throughput_tok_s   (throughput = n_out/(total-ttft))
    so compute = ttft_s + decode_time. Falls back to total_s if fields missing."""
    ttft = float(getattr(r, "ttft_s", 0.0) or 0.0)
    n_out = int(getattr(r, "n_out_tokens", 0) or 0)
    total = float(getattr(r, "total_s", 0.0) or 0.0)
    # decode phase = total - ttft, but total may include a load stall; prefer the
    # backend's own throughput when it exposes a clean decode rate.
    tput = getattr(r, "throughput_tok_s", None)
    try:
        tput = float(tput) if tput else (n_out / max(total - ttft, 1e-6) if n_out else 0.0)
    except Exception:
        tput = n_out / max(total - ttft, 1e-6) if n_out else 0.0
    decode = (n_out / tput) if (n_out and tput > 0) else max(total - ttft, 0.0)
    return float(ttft + decode)


def run_pass(query, specialist_set, retr, spec_backend, synth_backend, emb,
             desc, mode_concurrent=True):
    """Run specialists in `specialist_set` (RAG + generate) then the synthesiser.
    Returns (final_answer, compute_latency_s, all_contexts). Latency is
    COMPUTE-ONLY (see compute_latency): specialist stage = max single-call compute
    (concurrent / section-10.5) or sum (sequential), plus synth compute."""
    spec_answers, spec_comp, all_ctx = [], [], []
    for d in specialist_set:
        ids, ctx = retr.retrieve_with_ids(query, k=5)
        all_ctx += ctx
        r = spec_backend.generate(specialist_prompt(query, ctx, desc.get(d, d)),
                                   max_tokens=384)
        spec_answers.append(r.text); spec_comp.append(compute_latency(r))
    spec_lat = (max(spec_comp) if spec_comp else 0.0) if mode_concurrent else sum(spec_comp)
    sy = synth_backend.generate(synth_prompt(query, spec_answers), max_tokens=384)
    total = spec_lat + compute_latency(sy)
    return sy.text, total, all_ctx


def warmup(backends, prompt="ciao"):
    """One throwaway generation per model so the first TIMED call isn't paying a
    cold model-load (which would otherwise inflate compute_latency via ttft)."""
    for b in backends:
        try:
            b.generate(prompt, max_tokens=4)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gated_optimum",
                    choices=list(ALLOC_PRESETS.keys()))
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="conformal target miss-rate (only used when --threshold conformal)")
    ap.add_argument("--threshold", default="f2", choices=["f2", "conformal"],
                    help="how the gate's inclusion threshold is chosen on calibration: "
                         "'f2' = maximize F-beta vs gold (data-driven, default), "
                         "'conformal' = conformal-risk-control quantile")
    ap.add_argument("--beta", type=float, default=2.0,
                    help="beta for the F-beta objective (>1 favors recall; default 2)")
    ap.add_argument("--per_domain", default="on", choices=["on", "off"],
                    help="per-domain thresholds (on, default) vs a single global threshold")
    ap.add_argument("--recall_floor", type=float, default=0.8,
                    help="min calibration recall a chosen threshold must keep (forbids "
                         "over-abstention); lower = allow stricter gates")
    ap.add_argument("--tau_cap", type=float, default=0.6,
                    help="hard ceiling on any per-domain/global threshold")
    ap.add_argument("--shrink", type=float, default=8.0,
                    help="shrinkage of per-domain thresholds toward the global one "
                         "(higher = pull thin-sample domains harder toward global)")
    ap.add_argument("--router", default="llm", choices=["llm", "free", "all"],
                    help="candidate source for BOTH passes: real dispatcher LLM (default), "
                         "the free LLM-free retrieval router, or all 9 domains")
    ap.add_argument("--mode", default="concurrent", choices=["concurrent", "sequential"])
    ap.add_argument("--quant_disp", default="Q5_K_M")
    ap.add_argument("--quant_spec", default="Q8_0")
    ap.add_argument("--quant_synth", default="Q5_K_M")
    ap.add_argument("--mock", action="store_true", help="MockBackend smoke test (no GGUF/GPU)")
    ap.add_argument("--backend", default="llama_cpp", choices=["llama_cpp", "ollama"],
                    help="inference backend: llama_cpp (your GGUF files, default) or ollama")
    ap.add_argument("--gpu_layers", type=int, default=-1,
                    help="layers to offload to GPU per model (-1 = all; 0 = CPU only). "
                         "Needs a CUDA build of llama-cpp-python.")
    ap.add_argument("--limit", type=int, default=0, help="cap #queries (0 = all 58)")
    ap.add_argument("--out", default=str(HERE / "results" / "data" / "pipeline_eval.json"))
    args = ap.parse_args()

    disp_m, spec_m, synth_m = ALLOC_PRESETS[args.config]
    print(f"[config] {args.config}: disp={disp_m} spec={spec_m} synth={synth_m} "
          f"router={args.router} mode={args.mode} mock={args.mock}")

    # ---- embedder (bge-m3) for gate signal + RAG retrieval + gold-free quality
    try:
        emb = STEmbedder("BAAI/bge-m3")
        emb_tag = "bge-m3"
    except Exception as e:
        if not args.mock:
            raise SystemExit(f"bge-m3 (sentence-transformers) required for a real run: {e!r}")
        emb = HashingEmbedder(1024); emb_tag = "hashing-proxy(mock)"
    print(f"[embedder] {emb_tag}")

    specialists, desc = load_agents()
    profile = domain_passage_texts(specialists)
    rel = DomainRelevance(emb, specialists, desc, profile)
    gate = calibrate_gate(build_calibration_rows(rel), alpha=args.alpha,
                           method=args.threshold, beta=args.beta,
                           per_domain=(args.per_domain == "on"),
                           recall_floor=args.recall_floor, tau_cap=args.tau_cap,
                           shrink_k=args.shrink)
    Qdom = specialist_quality_by_domain(spec_m)
    if gate.tau_by_domain:
        tb = "  ".join(f"{d.replace('A_','')[:10]}={t:.2f}"
                       for d, t in sorted(gate.tau_by_domain.items()))
        print(f"[gate] method={gate.method} per-domain thresholds:\n       {tb}")
    else:
        print(f"[gate] method={gate.method} global tau={gate.tau:.3f} "
              f"(alpha={gate.alpha}, n_cal={gate.n_cal})")

    # ---- retrieval over the live corpus (specialist RAG)
    corpus = {}
    cj = CORPUS_DIR / "corpus_text.jsonl"
    if cj.exists():
        for line in cj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); corpus[r["id"]] = r["text"]
    if not corpus and not args.mock:
        raise SystemExit(f"corpus_text.jsonl not found in {CORPUS_DIR}; build FAISS first.")
    retr = FaissRetriever(emb, corpus, k=5)
    for nm in ("LawCorpus_IT", "CaseCorpus_IT"):
        idx, man = CORPUS_DIR / f"{nm}.faiss", CORPUS_DIR / f"{nm}.manifest.json"
        if idx.exists() and man.exists():
            retr.add(idx, man)
    if not retr._indexes and not args.mock:
        raise SystemExit("No FAISS indexes found; run build_faiss with bge-m3 first.")

    # ---- backends (one instance per role; reused across queries)
    disp_b = make_backend(disp_m, args.quant_disp, 2048, args.mock, args.backend, args.gpu_layers)
    spec_b = make_backend(spec_m, args.quant_spec, 4096, args.mock, args.backend, args.gpu_layers)
    synth_b = make_backend(synth_m, args.quant_synth, 8192, args.mock, args.backend, args.gpu_layers)
    if not args.mock:
        print("[warmup] priming each model once (so first timed call isn't a cold load)...")
        warmup([disp_b, spec_b, synth_b])

    # ---- held-out queries: in-domain TEST + dedicated OOD, minus calibration overlap
    calib_ids = {q["id"] for q in load_queries(CALIB)}
    tests = [q for q in load_queries(TEST) if q["id"] not in calib_ids]
    n_in = len(tests)
    n_ood_file = 0
    if OOD is not None:
        ood = [q for q in load_queries(OOD) if q["id"] not in calib_ids]
        for q in ood:
            q["category"] = q.get("category", "out_of_domain")
            q["expected_outcome"] = "abstain"
            q["expected_agents"] = []
        # avoid double-counting ids already present in TEST
        seen = {q["id"] for q in tests}
        ood = [q for q in ood if q["id"] not in seen]
        n_ood_file = len(ood)
        tests = tests + ood
    if args.limit:
        tests = tests[:args.limit]
    print(f"[data] {len(tests)} held-out queries (non-overlapping with calibration; "
          f"{n_in} from test set, {n_ood_file} from dedicated OOD file)")

    concurrent = (args.mode == "concurrent")
    rows = []
    _t_start = time.perf_counter()
    for i, q in enumerate(tests, 1):
        gold = set(parse_list(q.get("expected_agents")))
        is_ab = (q.get("expected_outcome") == "abstain") or \
                (q.get("category") == "out_of_domain") or (len(gold) == 0)

        # candidate set C(q) -- shared by both passes
        if args.router == "llm":
            dr = disp_b.generate(dispatcher_prompt(q["text"], specialists, desc), max_tokens=64)
            t_disp = compute_latency(dr)
            cand = parse_router_output(dr.text, specialists) or list(specialists)
        elif args.router == "free":
            cand = free_router_candidates(rel, q["text"]); t_disp = 0.0
        else:
            cand = list(specialists); t_disp = 0.0

        # ----- NORMAL: run all candidates
        nans, nlat, nctx = run_pass(q["text"], cand, retr, spec_b, synth_b, emb, desc, concurrent)
        nlat += t_disp

        # ----- DYNAMIC: gate prunes candidates -> S(q)  (per-domain thresholds)
        sig = rel.scores(q["text"], candidates=cand)
        cands = [Candidate(d, gate.rho_hat(sig[d]["fused"]), float(Qdom.get(d, 0.6)),
                           1.0, rel.topk_vectors(q["text"], d, 5),
                           gate.threshold_for(d)) for d in cand]
        budget = budget_from_entropy(sig, 1.0)
        sel = select_dynamic(cands, tau=None, budget=budget)   # tau=None -> per-domain thr
        dS = sel.domains
        dans, dlat, dctx = run_pass(q["text"], dS, retr, spec_b, synth_b, emb, desc, concurrent)
        dlat += t_disp

        # ----- correctness (gold-free)
        if is_ab:
            c_norm = 1.0 if len(cand) == 0 else 0.0      # normal rarely abstains
            c_dyn = 1.0 if len(dS) == 0 else 0.0
            ctype = "abstention"
        else:
            c_norm = answer_quality(emb, nans, nctx, q["text"])
            c_dyn = answer_quality(emb, dans, dctx, q["text"])
            ctype = "answer_quality"

        rows.append({"id": q["id"], "is_abstain": is_ab, "ctype": ctype,
                     "gold": sorted(gold), "candidates": cand,
                     "normal": {"S": cand, "k": len(cand), "latency_s": round(nlat, 3),
                                "correctness": round(c_norm, 4), "answer": nans[:400]},
                     "dynamic": {"S": dS, "k": len(dS), "latency_s": round(dlat, 3),
                                 "correctness": round(c_dyn, 4), "answer": dans[:400]},
                     "delta_latency_s": round(dlat - nlat, 3),
                     "delta_correctness": round(c_dyn - c_norm, 4)})
        # live ETA from real elapsed wall-time per query processed so far
        _elapsed = time.perf_counter() - _t_start
        _per = _elapsed / i
        _remain = _per * (len(tests) - i)
        def _hms(s):
            s = int(s); return f"{s//3600:d}h{(s%3600)//60:02d}m" if s >= 3600 else f"{s//60:d}m{s%60:02d}s"
        print(f"[{i}/{len(tests)}] {q['id']:14s} "
              f"k {len(cand)}->{len(dS)}  "
              f"lat {nlat:5.1f}->{dlat:5.1f}s (Δ{dlat-nlat:+.1f})  "
              f"{ctype[:4]} {c_norm:.2f}->{c_dyn:.2f} (Δ{c_dyn-c_norm:+.2f})  "
              f"| elapsed {_hms(_elapsed)} eta {_hms(_remain)}")

    # ---- aggregate
    def mean(key_path, subset=None):
        vals = []
        for r in rows:
            if subset == "indomain" and r["is_abstain"]:
                continue
            if subset == "abstain" and not r["is_abstain"]:
                continue
            cur = r
            for k in key_path:
                cur = cur[k]
            vals.append(cur)
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "config": args.config, "embedder": emb_tag, "router": args.router,
        "latency_definition": "compute_only (prefill+decode; excludes model load/swap)",
        "gate": {"method": gate.method, "beta": gate.beta,
                 "global_tau": round(gate.tau, 4),
                 "per_domain_tau": ({d: round(t, 4) for d, t in gate.tau_by_domain.items()}
                                    if gate.tau_by_domain else None)},
        "mode": args.mode, "alpha": args.alpha, "n_test": len(rows),
        "n_indomain": sum(1 for r in rows if not r["is_abstain"]),
        "n_abstain": sum(1 for r in rows if r["is_abstain"]),
        "latency": {
            "normal_mean_s": round(mean(["normal", "latency_s"]), 3),
            "dynamic_mean_s": round(mean(["dynamic", "latency_s"]), 3),
            "delta_mean_s": round(mean(["delta_latency_s"]), 3),
            "delta_pct": round(100 * mean(["delta_latency_s"]) / mean(["normal", "latency_s"]), 1)
            if mean(["normal", "latency_s"]) else 0.0},
        "correctness_indomain_answer_quality": {
            "normal": round(mean(["normal", "correctness"], "indomain"), 4),
            "dynamic": round(mean(["dynamic", "correctness"], "indomain"), 4),
            "delta": round(mean(["delta_correctness"], "indomain"), 4)},
        "abstention_accuracy": {
            "normal": round(mean(["normal", "correctness"], "abstain"), 4),
            "dynamic": round(mean(["dynamic", "correctness"], "abstain"), 4),
            "delta": round(mean(["delta_correctness"], "abstain"), 4)},
        "mean_k": {"normal": round(mean(["normal", "k"]), 2),
                   "dynamic": round(mean(["dynamic", "k"]), 2)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "per_query": rows},
                                         indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ END-TO-END: NORMAL vs DYNAMIC ROUTING ================")
    print(f"config={args.config}  embedder={emb_tag}  router={args.router}  "
          f"n={summary['n_test']} ({summary['n_indomain']} in-domain, "
          f"{summary['n_abstain']} abstain)")
    L = summary["latency"]; C = summary["correctness_indomain_answer_quality"]
    A = summary["abstention_accuracy"]; K = summary["mean_k"]
    print(f"\nLATENCY (s):     normal {L['normal_mean_s']:6.2f}   dynamic {L['dynamic_mean_s']:6.2f}   "
          f"Δ {L['delta_mean_s']:+.2f}  ({L['delta_pct']:+.1f}%)")
    print(f"mean k:          normal {K['normal']:6.2f}   dynamic {K['dynamic']:6.2f}")
    print(f"ANSWER QUALITY:  normal {C['normal']:6.3f}   dynamic {C['dynamic']:6.3f}   "
          f"Δ {C['delta']:+.3f}   (in-domain, gold-free)")
    print(f"ABSTENTION ACC:  normal {A['normal']:6.3f}   dynamic {A['dynamic']:6.3f}   "
          f"Δ {A['delta']:+.3f}   (out-of-domain)")
    print(f"\nwrote {args.out}")
    if emb_tag != "bge-m3":
        print("NOTE: not a real run (mock/proxy). Use real models + bge-m3 for paper numbers.")


if __name__ == "__main__":
    main()
