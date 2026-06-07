"""
part1_allocation/measure/routing_report.py
==========================================
Reconstruct the dispatcher's routing metrics from the quality table alone
(everything needed was logged: predicted agents in `output`, gold in
`expected_agents`). Reports multi-label P/R/F1 and exact-match per dispatcher
config, and a breakdown by difficulty -- useful as a paper figure and as the
empirical basis for Part 2's difficulty-aware policy.
"""
from __future__ import annotations

import pandas as pd

from part1_allocation.scoring.scorer import multilabel_prf


def _split(s) -> set[str]:
    if not isinstance(s, str) or not s:
        return set()
    return {x for x in s.split("|") if x}


def routing_report(quality_df: pd.DataFrame, dispatcher_id: str = "A_dispatcher"
                   ) -> pd.DataFrame:
    """Per (config, difficulty) routing metrics for the dispatcher rows."""
    df = quality_df[quality_df["agent"] == dispatcher_id].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for cid, g_cfg in df.groupby("config_id"):
        for diff, g in list(g_cfg.groupby("difficulty")) + [("__all__", g_cfg)]:
            ps, rs, fs, exact = [], [], [], []
            for _, row in g.iterrows():
                pred = _split(row["output"])
                exp = _split(row["expected_agents"])
                p, r, f1 = multilabel_prf(pred, exp)
                ps.append(p); rs.append(r); fs.append(f1)
                exact.append(1.0 if pred == exp else 0.0)
            n = len(g)
            rows.append({
                "config_id": cid, "difficulty": diff, "n": n,
                "precision": sum(ps) / n, "recall": sum(rs) / n,
                "f1": sum(fs) / n, "exact_match": sum(exact) / n,
            })
    out = pd.DataFrame(rows).sort_values(["config_id", "difficulty"])
    return out.reset_index(drop=True)


def print_routing_summary(quality_df: pd.DataFrame, dispatcher_id: str = "A_dispatcher"):
    rep = routing_report(quality_df, dispatcher_id)
    if rep.empty:
        print("[routing] no dispatcher rows (pass --calib to evaluate routing).")
        return
    overall = rep[rep["difficulty"] == "__all__"].sort_values("f1", ascending=False)
    print("[routing] dispatcher F1 by config (overall):")
    for _, r in overall.iterrows():
        print(f"   {r['config_id']:<34} F1={r['f1']:.3f}  "
              f"P={r['precision']:.3f} R={r['recall']:.3f}  exact={r['exact_match']:.3f}")
