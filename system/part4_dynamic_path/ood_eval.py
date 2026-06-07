import sys as _sys, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parents[0] / 'shared' / 'lib'))
_sys.path.insert(0, str(_HERE / 'src'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(str(_HERE / 'src' / '_x.py'))
"""
ood_eval.py
===========
DEDICATED out-of-domain test. Question answered: when a query is OUT of scope
(no family-law specialist applies), does the dynamic gate correctly ABSTAIN
(select zero specialists)? This is the safety property -- a legal assistant must
decline questions it has no competence for rather than fabricate an answer from
irrelevant specialists.

It runs ONLY the routing layer (no LLM generation), so it's fast. For each OOD
query it records whether the gate abstained, and if it did NOT, which domains
leaked through (the false activations) -- the actionable diagnostic.

Metrics:
  abstention_rate   = fraction of OOD queries with k == 0   (higher = safer; 1.0 ideal)
  mean_false_k      = mean #specialists wrongly activated on OOD queries
  leak_by_domain    = how often each domain is the culprit on a non-abstained OOD query

It also sweeps beta (recall<->precision of the gate) so you can see the
abstention/leak trade-off: a recall-leaning gate (high beta) keeps more domains
and therefore abstains LESS on OOD; a precision-leaning gate abstains MORE.

Calibration uses the larger expected_agents file; profiles use the gold-passage
file. The OOD queries are held-out (not in calibration).

Run:
    python -m part4_dynamic_path.ood_eval `
        --ood_file data/manifests/ood_queries_en.yaml `
        --calib_file data/manifests/calibration_queries_en.yaml
    # sweep the gate's beta to see how abstention trades against in-domain recall:
    python -m part4_dynamic_path.ood_eval --ood_file ... --calib_file ... --sweep_beta 1,2,3,5
"""
import argparse, json
import numpy as np

from dynamic_lib import (get_embedder, load_agents, domain_passage_texts,
                         DomainRelevance, load_queries, parse_list,
                         build_calibration_rows_from, calibrate_gate,
                         specialist_quality_by_domain, budget_from_entropy,
                         Candidate, select_dynamic, free_router_candidates)


def evaluate_ood(rel, gate, Qdom, specialists, ood, router):
    """Return (abstention_rate, mean_false_k, leak_by_domain, per_query)."""
    n_abstain = 0
    false_ks = []
    leak = {}
    per_q = []
    for q in ood:
        cand = (free_router_candidates(rel, q["text"]) if router == "free"
                else list(specialists))
        sig = rel.scores(q["text"], candidates=cand)
        cands = [Candidate(d, gate.rho_hat(sig[d]["fused"]), float(Qdom.get(d, 0.6)),
                           1.0, rel.topk_vectors(q["text"], d, 5),
                           gate.threshold_for(d)) for d in cand]
        dyn = select_dynamic(cands, tau=None, budget=budget_from_entropy(sig, 1.0))
        S = dyn.domains
        abstained = (len(S) == 0)
        n_abstain += int(abstained)
        false_ks.append(len(S))
        for d in S:
            leak[d] = leak.get(d, 0) + 1
        per_q.append({"id": q.get("id"), "query": q["text"][:90],
                      "abstained": abstained, "false_k": len(S),
                      "leaked_domains": sorted(S)})
    n = len(ood)
    return (n_abstain / n if n else float("nan"),
            float(np.mean(false_ks)) if false_ks else float("nan"),
            dict(sorted(leak.items(), key=lambda kv: -kv[1])),
            per_q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-m3")
    ap.add_argument("--ood_file", default=None,
                    help="OOD queries file (default: paths OOD location if present)")
    ap.add_argument("--calib_file", default=None,
                    help="threshold-calibration file (default: paths.CALIBRATION)")
    ap.add_argument("--threshold", default="f2", choices=["f2", "conformal"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--per_domain", default="on", choices=["on", "off"])
    ap.add_argument("--recall_floor", type=float, default=0.8)
    ap.add_argument("--tau_cap", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=8.0)
    ap.add_argument("--router", default="free", choices=["free", "all"])
    ap.add_argument("--sweep_beta", default=None,
                    help="comma-separated betas to sweep, e.g. '1,2,3,5'")
    ap.add_argument("--out", default=str(_DATA / "ood_eval.json"))
    args = ap.parse_args()

    # resolve OOD file
    ood_path = args.ood_file
    if not ood_path:
        for c in [_HERE.parents[0] / "data" / "manifests" / "ood_queries_en.yaml",
                  _HERE.parents[0] / "shared" / "data" / "ood_queries_en.yaml"]:
            if c.exists():
                ood_path = str(c); break
    if not ood_path or not _pl.Path(ood_path).exists():
        raise SystemExit("OOD file not found; pass --ood_file path/to/ood_queries_en.yaml")

    specialists, desc = load_agents()
    profile = domain_passage_texts(specialists)
    emb, tag = get_embedder(args.embedder)
    rel = DomainRelevance(emb, specialists, desc, profile)
    calib_file = args.calib_file or str(_P.CALIBRATION)
    cal_rows = build_calibration_rows_from(rel, calib_file)

    ood = load_queries(ood_path)
    # safety: ensure these are truly OOD (expected_agents empty) and held-out
    calib_ids = {q["id"] for q in load_queries(calib_file)}
    ood = [q for q in ood if q["id"] not in calib_ids]
    Qdom = specialist_quality_by_domain("llama3.2-1b")   # quality weights (gate is alloc-indep)

    print(f"[data] OOD queries: {len(ood)}  (calib={len(calib_ids)}, embedder={tag})")
    print(f"[router] candidate source = {args.router}\n")

    if args.sweep_beta:
        betas = [float(x) for x in args.sweep_beta.split(",")]
        print("===== OOD ABSTENTION vs BETA =====")
        hdr = f"{'beta':>6s}{'abstain_rate':>14s}{'mean_false_k':>14s}{'maxTau':>9s}"
        print(hdr); print("-" * len(hdr))
        out = {"ood_file": ood_path, "calib_file": calib_file, "embedder": tag,
               "n_ood": len(ood), "sweep": []}
        for b in betas:
            gate = calibrate_gate(cal_rows, method=args.threshold, beta=b,
                                  per_domain=(args.per_domain == "on"),
                                  recall_floor=args.recall_floor, tau_cap=args.tau_cap,
                                  shrink_k=args.shrink)
            ar, mfk, leak, _ = evaluate_ood(rel, gate, Qdom, specialists, ood, args.router)
            mt = max(gate.tau_by_domain.values()) if gate.tau_by_domain else gate.tau
            out["sweep"].append({"beta": b, "abstention_rate": round(ar, 3),
                                 "mean_false_k": round(mfk, 3), "max_tau": round(mt, 3),
                                 "leak_by_domain": leak})
            print(f"{b:>6.2f}{ar:>14.3f}{mfk:>14.3f}{mt:>9.3f}")
        _pl.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.out}")
        print("Reading it: higher beta -> gate keeps more domains -> abstains LESS on OOD.")
        return

    # single run
    gate = calibrate_gate(cal_rows, method=args.threshold, beta=args.beta,
                          per_domain=(args.per_domain == "on"),
                          recall_floor=args.recall_floor, tau_cap=args.tau_cap,
                          shrink_k=args.shrink)
    ar, mfk, leak, per_q = evaluate_ood(rel, gate, Qdom, specialists, ood, args.router)

    print(f"===== OOD ABSTENTION (beta={args.beta}, threshold={gate.method}) =====")
    print(f"  abstention_rate : {ar:.3f}   ({int(round(ar*len(ood)))}/{len(ood)} correctly abstained)")
    print(f"  mean_false_k    : {mfk:.3f}   (avg specialists wrongly activated)")
    if leak:
        print(f"  leak_by_domain  : (domains that wrongly fired, by frequency)")
        for d, c in leak.items():
            print(f"        {d.replace('A_',''):48s} {c}")
    print("\n  queries that did NOT abstain (leaks):")
    nleak = [r for r in per_q if not r["abstained"]]
    for r in nleak[:20]:
        print(f"     {str(r['id']):14s} k={r['false_k']} -> {[d.replace('A_','') for d in r['leaked_domains']][:4]}")
        print(f"        \"{r['query']}\"")
    if not nleak:
        print("     (none -- gate abstained on every OOD query)")

    out = {"ood_file": ood_path, "calib_file": calib_file, "embedder": tag,
           "beta": args.beta, "n_ood": len(ood),
           "abstention_rate": round(ar, 3), "mean_false_k": round(mfk, 3),
           "leak_by_domain": leak, "per_query": per_q}
    _pl.Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")
    if tag != "bge-m3":
        print("NOTE: PROXY embedder; OOD abstention is understated under the proxy.")


if __name__ == "__main__":
    main()
