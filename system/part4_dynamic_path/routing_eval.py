import sys as _sys, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parents[0] / 'shared' / 'lib'))
_sys.path.insert(0, str(_HERE / 'src'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(str(_HERE / 'src' / '_x.py'))
"""
06_routing_eval.py
==================
ROUTING-ONLY evaluation of normal vs dynamic routing -- NO answer-quality, NO LLM
generation. Reports, per query and as means:
  * routing F1 (and precision/recall) vs gold expected_agents
  * latency from the MEASURED perf_table via the chosen k (consistent with the
    Parts 1-3 frontier), and its delta

Because there is no generation, this runs in SECONDS on CPU (only bge-m3 for the
gate signal). Compare against the live end-to-end run for the realized number;
this gives the perf-table number that sits on the same axis as Parts 1-3.

Thresholds are calibrated on a LARGER expected_agents set (default: the 94-query
calibration file) for stable per-domain F2 thresholds, while the relevance
PROFILES stay built from the gold-passage file (which the larger file lacks).
Test queries are filtered to be non-overlapping with that calibration set.

Run (CPU is fine; bge-m3 needed for real numbers):
    python -m part4_dynamic_path.06_routing_eval --calib_file data/manifests/calibration_queries_en.yaml
    python -m part4_dynamic_path.06_routing_eval --embedder hashing      # proxy smoke test
"""
import argparse, json
import numpy as np

from dynamic_lib import (get_embedder, load_agents, domain_passage_texts,
                         DomainRelevance, load_queries, parse_list,
                         build_calibration_rows_from, calibrate_gate,
                         specialist_quality_by_domain, cost_model,
                         budget_from_entropy, Candidate, select_dynamic,
                         free_router_candidates, ALLOC_PRESETS)


def prf(pred, gold):
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred:
        return 1.0, 0.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0
    tp = len(pred & gold)
    p, r = tp / len(pred), tp / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-m3")
    ap.add_argument("--calib_file", default=None,
                    help="threshold-calibration file (expected_agents). Default: paths.CALIBRATION")
    ap.add_argument("--threshold", default="f2", choices=["f2", "conformal"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--per_domain", default="on", choices=["on", "off"])
    ap.add_argument("--recall_floor", type=float, default=0.8)
    ap.add_argument("--tau_cap", type=float, default=0.6)
    ap.add_argument("--shrink", type=float, default=8.0)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--router", default="free", choices=["free", "all"],
                    help="candidate source (no LLM here): free retrieval router or all 9")
    ap.add_argument("--mode", default="concurrent", choices=["concurrent", "sequential"])
    ap.add_argument("--sweep_recall_floor", default=None,
                    help="comma-separated recall floors to sweep, e.g. '0.7,0.8,0.9,1.0'. "
                         "Reports F1/recall/precision/k/latency per floor for ONE config "
                         "(--sweep_config), to expose the precision-recall trade-off.")
    ap.add_argument("--sweep_beta", default=None,
                    help="comma-separated F-beta values to sweep, e.g. '1,2,3,5'. "
                         "beta is the real precision/recall knob: low beta = precision-"
                         "leaning (higher threshold, fewer domains), high beta = recall-"
                         "leaning. Reports the trade-off per beta for --sweep_config.")
    ap.add_argument("--sweep_config", default="gated_optimum", choices=list(ALLOC_PRESETS.keys()))
    ap.add_argument("--out", default=str(_DATA / "routing_eval.json"))
    args = ap.parse_args()

    specialists, desc = load_agents()
    profile = domain_passage_texts(specialists)      # PROFILES from gold-passage file
    emb, tag = get_embedder(args.embedder)
    rel = DomainRelevance(emb, specialists, desc, profile)
    # optional frozen score cache (built on a bge-m3 machine via
    # scripts/build_score_cache.py): replays exact scores so this reproduces the
    # paper offline / without a GPU. Absent -> live embedder is used as before.
    _cache_path = _HERE / "data" / "score_cache.json"
    if _cache_path.exists():
        import json as _json
        _cj = _json.loads(_cache_path.read_text(encoding="utf-8"))
        rel.score_cache = _cj.get("scores", {})
        print(f"[cache] loaded {len(rel.score_cache)} frozen query scores "
              f"(embedder={_cj.get('embedder','?')}) from score_cache.json")
        if rel.score_cache and _cj.get("embedder") == "bge-m3":
            # the scores actually used are replayed from the frozen bge-m3 cache,
            # so the run is a bge-m3 run regardless of the live-embedder fallback
            tag = "bge-m3 (cached)"
    # default calibration = the 94-query routing-annotated set if present, else the
    # 25-query gold-text file (paper §12: thresholds are fit on the 94-query set).
    _default_calib = getattr(_P, "CALIBRATION_ROUTING", _P.CALIBRATION)
    if not _pl.Path(str(_default_calib)).exists():
        _default_calib = _P.CALIBRATION
    calib_file = args.calib_file or str(_default_calib)
    cal_rows = build_calibration_rows_from(rel, calib_file)
    calib_ids = {q["id"] for q in load_queries(calib_file)}
    tests = [q for q in load_queries(_P.TEST_QUERIES) if q["id"] not in calib_ids]

    # ---------- recall-floor sweep mode ----------
    if args.sweep_recall_floor or args.sweep_beta:
        is_beta = bool(args.sweep_beta)
        values = [float(x) for x in (args.sweep_beta if is_beta
                                     else args.sweep_recall_floor).split(",")]
        pname = "beta" if is_beta else "rfloor"
        dm, sm, ym = ALLOC_PRESETS[args.sweep_config]
        Qdom = specialist_quality_by_domain(sm)
        cm = cost_model(dm, sm, ym, args.mode)
        nlat0 = cm.latency(6)
        kind = "BETA" if is_beta else "RECALL-FLOOR"
        print(f"\n===== {kind} SWEEP ({args.sweep_config}, calib={len(calib_ids)}, "
              f"test={len(tests)}, embedder={tag}) =====")
        hdr = (f"{pname:>7s}{'F1':>7s}{'prec':>7s}{'rec':>7s}{'mean_k':>8s}"
               f"{'lat(s)':>8s}{'-lat%':>7s}{'abst':>7s}{'maxTau':>8s}")
        print(hdr); print("-" * len(hdr))
        sweep = []
        for v in values:
            beta = v if is_beta else args.beta
            rfloor = args.recall_floor if is_beta else v
            gate = calibrate_gate(cal_rows, method=args.threshold, beta=beta,
                                  per_domain=(args.per_domain == "on"),
                                  recall_floor=rfloor, tau_cap=args.tau_cap, shrink_k=args.shrink)
            Ps, Rs, Fs, Ks, Ls, Abs = [], [], [], [], [], []
            for q in tests:
                gold = set(parse_list(q.get("expected_agents")))
                is_ab = (not gold) or q.get("expected_outcome") == "abstain" \
                    or q.get("category") == "out_of_domain"
                cand = (free_router_candidates(rel, q["text"]) if args.router == "free"
                        else list(specialists))
                sig = rel.scores(q["text"], candidates=cand)
                cands = [Candidate(d, gate.rho_hat(sig[d]["fused"]), float(Qdom.get(d, 0.6)),
                                   1.0, rel.topk_vectors(q["text"], d, 5),
                                   gate.threshold_for(d)) for d in cand]
                dyn = select_dynamic(cands, tau=None, budget=budget_from_entropy(sig, 1.0))
                pred = set(dyn.domains)
                p, r, f = prf(pred, gold)
                Ps.append(p); Rs.append(r); Fs.append(f); Ks.append(len(pred))
                Ls.append(cm.latency(len(pred)))
                if is_ab:
                    Abs.append(1.0 if len(pred) == 0 else 0.0)
            mt = max(gate.tau_by_domain.values()) if gate.tau_by_domain else gate.tau
            mean = lambda x: float(np.mean(x)) if x else float("nan")
            row = {pname: v, "beta": beta, "recall_floor": rfloor,
                   "f1": round(mean(Fs), 3), "precision": round(mean(Ps), 3),
                   "recall": round(mean(Rs), 3), "mean_k": round(mean(Ks), 2),
                   "mean_latency_s": round(mean(Ls), 2),
                   "latency_vs_normal_pct": round(100 * (1 - mean(Ls) / nlat0), 1),
                   "abstain_accuracy": round(mean(Abs), 3) if Abs else None,
                   "max_tau": round(mt, 3)}
            sweep.append(row)
            ab = row["abstain_accuracy"]
            print(f"{v:>7.2f}{row['f1']:>7.2f}{row['precision']:>7.2f}{row['recall']:>7.2f}"
                  f"{row['mean_k']:>8.2f}{row['mean_latency_s']:>8.2f}"
                  f"{row['latency_vs_normal_pct']:>6.1f}%"
                  f"{(ab if ab is not None else float('nan')):>7.2f}{row['max_tau']:>8.3f}")
        fname = "routing_beta_sweep.json" if is_beta else "routing_recall_sweep.json"
        (_DATA / fname).write_text(
            json.dumps({"config": args.sweep_config, "swept": pname, "embedder": tag,
                        "calib_file": calib_file, "n_test": len(tests),
                        "sweep": sweep}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {_DATA / fname}")
        return

    gate = calibrate_gate(cal_rows,
                          alpha=args.alpha, method=args.threshold, beta=args.beta,
                          per_domain=(args.per_domain == "on"),
                          recall_floor=args.recall_floor, tau_cap=args.tau_cap,
                          shrink_k=args.shrink)
    if gate.tau_by_domain:
        print("[gate] per-domain thresholds (calib =", calib_file.split("/")[-1], "):")
        for d, t in sorted(gate.tau_by_domain.items()):
            print(f"        {d.replace('A_',''):48s} {t:.3f}")
        print(f"        max={max(gate.tau_by_domain.values()):.3f}")
    else:
        print(f"[gate] global tau={gate.tau:.3f} method={gate.method}")

    # held-out test queries, non-overlapping with the calibration file used
    calib_ids = {q["id"] for q in load_queries(calib_file)}
    tests = [q for q in load_queries(_P.TEST_QUERIES) if q["id"] not in calib_ids]
    n_ab = sum(1 for q in tests if not parse_list(q.get("expected_agents"))
               or q.get("expected_outcome") == "abstain"
               or q.get("category") == "out_of_domain")
    print(f"[data] {len(tests)} held-out test queries "
          f"({len(tests)-n_ab} in-domain, {n_ab} OOD); calib={len(calib_ids)} queries")

    concurrent = (args.mode == "concurrent")
    out = {"embedder": tag, "calib_file": calib_file, "n_test": len(tests),
           "gate_method": gate.method,
           "per_domain_tau": (gate.tau_by_domain if gate.tau_by_domain else None),
           "presets": {}}

    for name, (dm, sm, ym) in ALLOC_PRESETS.items():
        Qdom = specialist_quality_by_domain(sm)
        cm = cost_model(dm, sm, ym, args.mode)
        agg = {sel: {k: [] for k in ("P", "R", "F", "k", "lat", "ab", "isab")}
               for sel in ("normal", "dynamic")}
        per_q = []
        for q in tests:
            gold = set(parse_list(q.get("expected_agents")))
            is_ab = (not gold) or q.get("expected_outcome") == "abstain" \
                or q.get("category") == "out_of_domain"
            # candidate set (no LLM): free retrieval router or all domains
            cand = (free_router_candidates(rel, q["text"]) if args.router == "free"
                    else list(specialists))
            sig = rel.scores(q["text"], candidates=cand)
            cands = [Candidate(d, gate.rho_hat(sig[d]["fused"]), float(Qdom.get(d, 0.6)),
                               1.0, rel.topk_vectors(q["text"], d, 5),
                               gate.threshold_for(d)) for d in cand]
            budget = budget_from_entropy(sig, 1.0)
            dyn = select_dynamic(cands, tau=None, budget=budget)
            sels = {"normal": set(cand), "dynamic": set(dyn.domains)}
            row = {"id": q["id"], "gold": sorted(gold), "is_abstain": is_ab}
            for s, pred in sels.items():
                p, r, f = prf(pred, gold)
                k = len(pred)
                a = agg[s]
                a["P"].append(p); a["R"].append(r); a["F"].append(f); a["k"].append(k)
                a["lat"].append(cm.latency(k)); a["isab"].append(is_ab)
                if is_ab:
                    a["ab"].append(1.0 if k == 0 else 0.0)
                row[s] = {"S": sorted(pred), "k": k, "lat_perftable_s": round(cm.latency(k), 3)}
            per_q.append(row)

        m = lambda x: float(np.mean(x)) if x else float("nan")
        nlat = m(agg["normal"]["lat"])
        rec = {"allocation": {"dispatcher": dm, "specialist": sm, "synthesiser": ym},
               "L_disp": round(cm.L_disp, 2), "L_spec_one": round(cm.L_spec, 2),
               "L_synth": round(cm.L_synth, 2)}
        for s in ("normal", "dynamic"):
            a = agg[s]
            rec[s] = {"precision": round(m(a["P"]), 3), "recall": round(m(a["R"]), 3),
                      "f1": round(m(a["F"]), 3), "mean_k": round(m(a["k"]), 2),
                      "mean_latency_s": round(m(a["lat"]), 2),
                      "latency_vs_normal_pct": round(100 * (1 - m(a["lat"]) / nlat), 1)
                      if nlat else 0.0,
                      "abstain_accuracy": round(m(a["ab"]), 3) if a["ab"] else None}
        rec["delta_f1"] = round(rec["dynamic"]["f1"] - rec["normal"]["f1"], 3)
        rec["delta_latency_s"] = round(rec["dynamic"]["mean_latency_s"]
                                       - rec["normal"]["mean_latency_s"], 2)
        out["presets"][name] = rec
        out["presets"][name]["_per_query"] = per_q if name == "gated_optimum" else "see gated_optimum"

    (_DATA / "routing_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ ROUTING-ONLY: NORMAL vs DYNAMIC (perf-table latency) ================")
    print(f"embedder={tag}  n_test={out['n_test']}  threshold={gate.method}")
    hdr = (f"{'preset':14s}{'sel':9s}{'F1':>6s}{'prec':>6s}{'rec':>6s}"
           f"{'k':>6s}{'lat(s)':>8s}{'-lat%':>7s}{'abst':>7s}")
    print(hdr); print("-" * len(hdr))
    for name in ALLOC_PRESETS:
        r = out["presets"][name]
        for s in ("normal", "dynamic"):
            d = r[s]; ab = d["abstain_accuracy"]
            print(f"{name:14s}{s:9s}{d['f1']:>6.2f}{d['precision']:>6.2f}{d['recall']:>6.2f}"
                  f"{d['mean_k']:>6.2f}{d['mean_latency_s']:>8.2f}{d['latency_vs_normal_pct']:>6.1f}%"
                  f"{(ab if ab is not None else float('nan')):>7.2f}")
        print(f"{'':14s}{'Δ':9s}{r['delta_f1']:>+6.2f}{'':>12s}{'':>6s}{r['delta_latency_s']:>+8.2f}")
    print(f"\nwrote {_DATA / 'routing_eval.json'}")
    if not tag.startswith("bge-m3"):
        print("NOTE: PROXY embedder; rerun with bge-m3 for real numbers.")


if __name__ == "__main__":
    main()
