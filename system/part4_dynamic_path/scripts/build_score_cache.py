"""
Build a FROZEN relevance-score cache so the dynamic-routing evaluation reproduces
the paper's bge-m3 numbers on a machine WITHOUT bge-m3 (or without a GPU).

RUN THIS ONCE on a machine where the real embedder loads (e.g. the RTX 4070 box):

    python -m part4_dynamic_path.scripts.build_score_cache --embedder bge-m3

It embeds every calibration + test query against the per-domain corpus profiles
and writes {query_text -> {domain -> {ret_max,ret_mean,desc,fused}}} to
part4_dynamic_path/data/score_cache.json. Thereafter routing_eval / ood_eval pick
the cache up automatically and REPLAY these exact scores with NO embedder call, so
the F1=0.50 / abstention=0.60 numbers are reproducible offline.

IMPORTANT: this computes REAL scores from the real embedder; it invents nothing.
Run it with --embedder bge-m3 (the proxy would cache weaker proxy scores, which are
clearly not the paper's result, and the script warns if you do).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_HERE = _pl.Path(__file__).resolve().parent          # .../part4_dynamic_path/scripts
_P4 = _HERE.parent                                   # .../part4_dynamic_path
_ROOT = _P4.parent                                   # repo root
_sys.path.insert(0, str(_ROOT / "shared" / "lib"))
_sys.path.insert(0, str(_P4 / "src"))
import paths as _P
import argparse, json, os
from dynamic_lib import (get_embedder, load_agents, domain_passage_texts,
                         DomainRelevance, load_queries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="bge-m3",
                    help="MUST be bge-m3 to reproduce the paper; the proxy caches "
                         "proxy scores (weaker) and is not the paper's result.")
    ap.add_argument("--out", default=str(_P4 / "data" / "score_cache.json"))
    ap.add_argument("--calib_file", default=None)
    ap.add_argument("--extra_files", default=None,
                    help="comma-separated extra query files to cache (e.g. the "
                         "OOD set), so ood_eval also replays offline.")
    args = ap.parse_args()
    os.environ["MAMAP_REBUILD_VECTORS"] = "1"   # builder must embed live

    specialists, desc = load_agents()
    profile = domain_passage_texts(specialists)       # PROFILES from the gold-passage file
    emb, tag = get_embedder(args.embedder)
    if "proxy" in tag.lower():
        print("WARNING: embedder is the hashing PROXY -- cached scores will NOT match "
              "the paper's bge-m3 numbers. Re-run with --embedder bge-m3.")
    rel = DomainRelevance(emb, specialists, desc, profile)

    calib_file = args.calib_file or str(getattr(_P, "CALIBRATION_ROUTING", _P.CALIBRATION))
    scores = {}
    srcs = [calib_file, str(_P.TEST_QUERIES)]
    if args.extra_files:
        srcs += [p for p in args.extra_files.split(",") if p]
    for src in srcs:
        for q in load_queries(src):
            txt = q.get("text") or q.get("query")
            if txt and txt not in scores:
                scores[txt] = rel.scores(txt)          # REAL scores from the real embedder

    _pl.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    _pl.Path(args.out).write_text(json.dumps(
        {"embedder": tag, "n_queries": len(scores), "scores": scores},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {len(scores)} queries cached (embedder={tag})")
    if "proxy" not in tag.lower():
        print("OK: cache is from the real embedder; routing_eval can reproduce the paper offline.")
        import numpy as _np
        vec_out = _P4 / "data" / "score_cache_vectors.npz"
        arrs = {"embedder": _np.array(tag)}
        for s in specialists:
            arrs[f"desc__{s}"] = rel._desc[s]
            arrs[f"mat__{s}"] = rel._mat[s]
        _np.savez_compressed(vec_out, **arrs)
        print(f"wrote {vec_out}: frozen domain-profile vectors (embedder={tag})")


if __name__ == "__main__":
    main()
