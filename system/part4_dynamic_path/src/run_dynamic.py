"""
part4_dynamic_path/src/run_dynamic.py
=====================================
End-to-end DYNAMIC AGENTIC-PATH selection over the held-out test queries.

Flow per query (respects the Parts 1-3 structure; no extra LLM):
  dispatcher -> candidate domains C(q)           [gold expected_agents as the
                                                  router proxy; swap in the live
                                                  dispatcher when running it]
  signals    -> per-domain retrieval+description relevance  (signals.py)
  calibrate  -> rho_hat_d, conformal tau                    (calibrate.py)
  select     -> S(q) by {dynamic, threshold, topk, full}    (select.py)
  cost       -> measured latency/energy of running S(q)      (costq.py)

We then EVALUATE each selector on the test set:
  * routing quality   : precision / recall / F1 of S(q) vs gold domains
  * abstain accuracy  : on the 15 out-of-domain queries (gold = []), does the
                        selector correctly choose S(q) = {} ?
  * cost              : mean latency / energy, and reduction vs FULL activation
  * endogenous k      : mean |S(q)|

This is a SELECTION evaluation -- it scores which domains are run. The final-
answer-quality evaluation (does pruning preserve the synthesised answer)
requires the synthesiser-subset re-run (separate script), which needs the real
models; this runner is deliberately model-free so it runs anywhere.

Embedder: tries bge-m3 (production) and falls back to the dependency-free
HashingEmbedder when the model/network is unavailable (the result is then a
PROXY -- clearly logged).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                      # repo root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from signals import (DomainRelevance, load_agents, load_corpus,        # noqa: E402
                     domain_passage_profile)
from calibrate import (calibrate_gate, build_calibration_signal_rows)  # noqa: E402
from selector import (Candidate, select_dynamic, select_threshold,       # noqa: E402
                    select_topk, select_full)
from costq import (specialist_quality_by_domain, cost_model_from_perf, # noqa: E402
                   budget_from_entropy)


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def get_embedder(prefer: str = "bge-m3"):
    """bge-m3 if available, else the offline HashingEmbedder (proxy)."""
    from part1_allocation.scoring.embeddings import HashingEmbedder
    if prefer and prefer.lower() != "hashing":
        try:
            from part1_allocation.scoring.embeddings import STEmbedder
            emb = STEmbedder(model_name="BAAI/bge-m3")
            print(f"[embedder] bge-m3 (production), dim={emb.dim}")
            return emb, "bge-m3"
        except Exception as e:
            print(f"[embedder] bge-m3 unavailable ({type(e).__name__}); "
                  f"falling back to HashingEmbedder PROXY.")
    emb = HashingEmbedder(1024)
    print(f"[embedder] HashingEmbedder PROXY, dim={emb.dim}")
    return emb, "hashing-proxy"


# --------------------------------------------------------------------------- metrics
def prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0          # correct abstention
    if not pred:
        return 1.0, 0.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0          # predicted something on an abstain query
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-m3",
                    help="'bge-m3' (default) or 'hashing' to force the proxy")
    ap.add_argument("--alpha", type=float, default=0.10,
                    help="conformal target miss-rate for relevant domains")
    ap.add_argument("--spec_model", default="llama3.2-1b",
                    help="fixed specialist model from Part-2 optimum")
    ap.add_argument("--disp_config", default="mistral-7b__Q5_K_M__c2048")
    ap.add_argument("--spec_config", default="llama3.2-1b__Q8_0__c4096")
    ap.add_argument("--synth_config", default="mistral-7b__Q5_K_M__c8192")
    ap.add_argument("--mode", default="concurrent",
                    choices=["concurrent", "sequential"])
    ap.add_argument("--use_router_candidates", action="store_true",
                    help="prune within gold expected_agents (router proxy). If "
                         "off, candidates = all 9 domains (pure retrieval gate).")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "dynamic_eval.json"))
    args = ap.parse_args()

    calib_yaml = REPO / "part1_allocation/data/calib_clean_with_gold_text.yaml"
    test_yaml = HERE.parent / "data" / "test_queries_en.yaml"
    corpus = load_corpus(REPO / "shared/corpus/corpus_text.jsonl")
    specialists, desc = load_agents(REPO / "part1_allocation/config/agents.yaml")
    profile = domain_passage_profile(calib_yaml, specialists)

    emb, emb_tag = get_embedder(args.embedder)
    rel = DomainRelevance(emb, corpus, specialists, desc, profile)

    # ---- offline calibration on the calibration set
    cal_rows = build_calibration_signal_rows(rel, calib_yaml)
    gate = calibrate_gate(cal_rows, alpha=args.alpha)
    print(f"[calibrate] conformal tau={gate.tau:.3f} (alpha={gate.alpha}, "
          f"n_relevant_cal={gate.n_cal})")

    # ---- measured per-domain specialist quality + cost model
    Qdom = specialist_quality_by_domain(HERE.parent / "data/quality_table.parquet",
                                        args.spec_model)
    perf = REPO / "shared/pareto/perf_table.parquet"
    try:
        cm = cost_model_from_perf(perf, args.disp_config, args.spec_config,
                                  args.synth_config, mode=args.mode)
    except KeyError:
        # configs not present -> pick representative ones by model prefix
        import pandas as pd
        df = pd.read_parquet(perf)
        def pick(modelpfx, ctx):
            cand = df[df.config_id.str.startswith(modelpfx) & df.config_id.str.contains(f"c{ctx}")]
            return cand.sort_values("peak_mem_gb").config_id.iloc[0]
        cm = cost_model_from_perf(perf, pick("mistral-7b", 2048),
                                  pick("llama3.2-1b", 4096),
                                  pick("mistral-7b", 8192), mode=args.mode)
    cost_one = cm.spec_cost_one("latency")
    print(f"[cost] L_disp={cm.L_disp:.2f}s L_spec(one)={cm.L_spec:.2f}s "
          f"L_synth={cm.L_synth:.2f}s mode={cm.mode}")

    # ---- run over test queries
    tests = yaml.safe_load(test_yaml.read_text(encoding="utf-8"))["queries"]
    selectors = ["full", "topk", "threshold", "dynamic"]
    agg = {s: {"P": [], "R": [], "F": [], "k": [], "lat": [], "ene": [],
               "abstain_ok": [], "is_abstain": []} for s in selectors}
    per_query = []

    for q in tests:
        gold = set(_parse(q.get("expected_agents")) or [])
        is_abstain = (len(gold) == 0)
        cset = list(specialists) if not args.use_router_candidates else \
            (list(gold) if gold else list(specialists))
        # NOTE: router-proxy candidates can't test abstention (gold drives them);
        # default (all domains) lets the gate itself decide to abstain.
        sig = rel.scores(q["text"], candidates=cset)
        cands = [
            Candidate(domain=d,
                      rho_hat=gate.rho_hat(sig[d]["fused"]),
                      quality=float(Qdom.get(d, 0.6)),
                      cost=cost_one,
                      vectors=rel.retrieved_vectors(q["text"], d, topk=5))
            for d in cset
        ]
        budget = budget_from_entropy(sig, cost_one)

        sels = {
            "full": select_full(cands),
            "topk": select_topk(cands, k=3),
            "threshold": select_threshold(cands, gate.tau),
            "dynamic": select_dynamic(cands, gate.tau, budget=budget),
        }
        row = {"id": q["id"], "gold": sorted(gold), "is_abstain": is_abstain}
        for s, sel in sels.items():
            pred = set(sel.domains)
            p, r, f = prf(pred, gold)
            agg[s]["P"].append(p); agg[s]["R"].append(r); agg[s]["F"].append(f)
            agg[s]["k"].append(sel.k)
            agg[s]["lat"].append(cm.latency(sel.k))
            agg[s]["ene"].append(cm.energy(sel.k))
            agg[s]["is_abstain"].append(is_abstain)
            if is_abstain:
                agg[s]["abstain_ok"].append(1.0 if sel.k == 0 else 0.0)
            row[s] = {"S": sel.domains, "k": sel.k,
                      "lat": round(cm.latency(sel.k), 3)}
        per_query.append(row)

    # ---- summarize
    def m(x): return float(np.mean(x)) if x else float("nan")
    full_lat = m(agg["full"]["lat"])
    summary = {"embedder": emb_tag, "alpha": args.alpha, "tau": gate.tau,
               "n_test": len(tests),
               "n_abstain": int(sum(agg["full"]["is_abstain"])),
               "selectors": {}}
    for s in selectors:
        a = agg[s]
        summary["selectors"][s] = {
            "precision": round(m(a["P"]), 3), "recall": round(m(a["R"]), 3),
            "f1": round(m(a["F"]), 3), "mean_k": round(m(a["k"]), 2),
            "mean_latency_s": round(m(a["lat"]), 2),
            "mean_energy_j": round(m(a["ene"]), 1),
            "latency_vs_full_pct": round(100 * (1 - m(a["lat"]) / full_lat), 1)
            if full_lat else 0.0,
            "abstain_accuracy": round(m(a["abstain_ok"]), 3) if a["abstain_ok"] else None,
        }

    Path(args.out).write_text(json.dumps(
        {"summary": summary, "per_query": per_query}, indent=2, ensure_ascii=False))
    print("\n================ DYNAMIC PATH SELECTION -- TEST SET ================")
    print(f"embedder={emb_tag}  n_test={summary['n_test']}  "
          f"abstain queries={summary['n_abstain']}  tau={gate.tau:.3f}")
    hdr = f"{'selector':10s} {'F1':>5s} {'prec':>5s} {'rec':>5s} {'mean_k':>7s} " \
          f"{'lat(s)':>7s} {'-lat%':>6s} {'abst.acc':>8s}"
    print(hdr); print("-" * len(hdr))
    for s in selectors:
        d = summary["selectors"][s]
        print(f"{s:10s} {d['f1']:>5.2f} {d['precision']:>5.2f} {d['recall']:>5.2f} "
              f"{d['mean_k']:>7.2f} {d['mean_latency_s']:>7.2f} "
              f"{d['latency_vs_full_pct']:>5.1f}% "
              f"{(d['abstain_accuracy'] if d['abstain_accuracy'] is not None else float('nan')):>8.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
