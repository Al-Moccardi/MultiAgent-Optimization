"""
part4_dynamic_path/src/synth_subset_rerun.py
============================================
THE one new experiment the dynamic part needs that cannot be done from existing
tables: does dropping a specialist actually change the SYNTHESISED final answer?

The quality table stores the synthesiser run on the FULL activated set only.
To (a) fit the value model's marginal/redundancy term and (b) *prove* that the
dynamic policy preserves answer quality, we must re-run ONLY the synthesiser on
SUBSETS of the already-generated specialist answers -- not the full pipeline.
The specialists' per-(query, domain) outputs already exist in the quality
table, so this re-run reuses everything expensive and only re-invokes the cheap
last hop.

Protocol (per in-domain calibration/test query with >=2 activated specialists):
  * FULL      : synthesise from all activated specialists' stored answers.
  * LOO       : leave-one-specialist-out -- synthesise from the rest; one per
                activated specialist. Measures each specialist's MARGINAL value.
  * DYNAMIC   : synthesise from exactly the set the dynamic selector chose.
Score each synthesised answer's correctness vs the gold answer (cosine of
bge-m3 embeddings, the project's correctness metric). The headline curve is
final-answer correctness vs |subset|, and the LOO drop tells you which
specialists are safe to prune.

This script REQUIRES the real environment (the synthesiser LLM via the project
backend + bge-m3). It is intentionally separate from run_dynamic.py (which is
model-free). Run it where Parts 1-3 were measured.

CAUTION: validate the synthesiser config produces coherent Italian output
before trusting the scores -- some stored synth outputs in early runs were
degraded boilerplate; if the FULL-set baseline is incoherent, fix the synth
prompt/config first or the whole comparison is measuring noise.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def load_specialist_outputs(quality_table: str | Path,
                            spec_model: str) -> dict[tuple[str, str], str]:
    """{(query_id, domain): specialist_answer_text} for the chosen specialist
    model, taken from the existing quality table (no generation)."""
    qt = pd.read_parquet(quality_table)
    qt = qt[~qt.agent.isin(["A_dispatcher", "A_synth"])].copy()
    qt["model"] = qt.config_id.apply(lambda c: str(c).split("__")[0])
    qt = qt[qt.model == spec_model]
    out: dict[tuple[str, str], str] = {}
    for _, r in qt.iterrows():
        out[(str(r["query_id"]), str(r["agent"]))] = str(r.get("output") or "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_model", default="llama3.2-1b")
    ap.add_argument("--synth_model", default="mistral-7b",
                    help="synthesiser model id used by the project backend")
    ap.add_argument("--quality_table",
                    default=str(HERE.parent / "data/quality_table.parquet"))
    ap.add_argument("--queries",
                    default=str(REPO / "part1_allocation/data/calib_clean_with_gold_text.yaml"))
    ap.add_argument("--max_subset_loo_only", action="store_true",
                    help="only leave-one-out subsets (cheap); else also small subsets")
    ap.add_argument("--out", default=str(HERE.parent / "results" / "synth_subset.jsonl"))
    args = ap.parse_args()

    # --- real backends (fail loudly if unavailable; this script needs them)
    from part1_allocation.scoring.embeddings import STEmbedder
    from part1_allocation.inference.backend import load_backend  # project's LLM backend
    emb = STEmbedder("BAAI/bge-m3")
    synth = load_backend(args.synth_model)   # <-- adapt to your backend API if named differently

    def synthesise(query: str, specialist_answers: list[str], contexts: list[str]) -> str:
        """Compose a final answer from specialist answers (+ contexts).
        Adapt the prompt to match the project's synthesiser prompt exactly."""
        joined = "\n\n".join(f"[Specialist {i+1}]\n{a}"
                             for i, a in enumerate(specialist_answers))
        prompt = (
            "You are the synthesiser of an Italian family-law assistant. "
            "Compose ONE coherent, correct final answer for the user, using only "
            "the specialist analyses below. Do not invent authorities.\n\n"
            f"User question:\n{query}\n\n{joined}\n\nFinal answer:"
        )
        return synth.generate(prompt, max_new_tokens=384)   # adapt to backend

    def correctness(answer: str, gold: str) -> float:
        a = emb.encode([answer])[0]; g = emb.encode([gold])[0]
        return float(max(0.0, np.dot(a, g) / (np.linalg.norm(a) * np.linalg.norm(g) + 1e-9)))

    spec_out = load_specialist_outputs(args.quality_table, args.spec_model)
    queries = yaml.safe_load(Path(args.queries).read_text(encoding="utf-8"))["queries"]

    fout = open(args.out, "w", encoding="utf-8")
    n_done = 0
    for q in queries:
        qid = str(q["id"])
        gold_doms = [d for d in (_parse(q.get("expected_agents")) or [])]
        gold_doms = [d for d in gold_doms if (qid, d) in spec_out]
        if len(gold_doms) < 2:
            continue
        gold_answer = str(q.get("ground_truth_answer") or "")
        # FULL
        full_ans = synthesise(q["text"], [spec_out[(qid, d)] for d in gold_doms], [])
        rec = {"qid": qid, "gold_domains": gold_doms,
               "full": {"k": len(gold_doms),
                        "correctness": correctness(full_ans, gold_answer)}}
        # LOO
        loo = {}
        for d in gold_doms:
            sub = [x for x in gold_doms if x != d]
            ans = synthesise(q["text"], [spec_out[(qid, s)] for s in sub], [])
            loo[d] = {"k": len(sub), "correctness": correctness(ans, gold_answer)}
        rec["loo"] = loo
        rec["loo_drop"] = {d: round(rec["full"]["correctness"] - loo[d]["correctness"], 4)
                           for d in gold_doms}
        # optional: all small subsets (k>=1) -- expensive, gated by flag
        if not args.max_subset_loo_only and len(gold_doms) <= 4:
            subsets = {}
            for r in range(1, len(gold_doms)):
                for combo in itertools.combinations(gold_doms, r):
                    ans = synthesise(q["text"], [spec_out[(qid, s)] for s in combo], [])
                    subsets["+".join(combo)] = round(correctness(ans, gold_answer), 4)
            rec["subsets"] = subsets
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        n_done += 1
        print(f"[{n_done}] {qid}: full={rec['full']['correctness']:.3f} "
              f"min_loo_drop={min(rec['loo_drop'].values()):.3f}")
    fout.close()
    print(f"\nwrote {args.out} ({n_done} multi-specialist queries)")
    print("Next: a specialist whose LOO drop ~ 0 across queries is safe to prune; "
          "fit the coverage/value weights to these drops, then re-run run_dynamic "
          "to score the dynamic policy's final-answer quality vs FULL.")


if __name__ == "__main__":
    main()
