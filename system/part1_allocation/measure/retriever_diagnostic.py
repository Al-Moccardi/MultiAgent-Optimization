"""
part1_allocation/measure/retriever_diagnostic.py
================================================
Compute retriever-only diagnostics (context_precision, context_recall) from the
specialist quality table, and write them to retriever_diagnostic.csv.

These metrics characterise the FAISS + embedder pipeline; they do NOT depend on
which candidate model serves the agent. We average across all specialist samples
(the contexts retrieved are the same for every candidate, so any (config_id, query)
pair contributes the same retrieval scores -- we just take the per-query value).

Output: one summary row per (corpus/run) plus a CSV with per-query breakdown.
This is reported in the paper separately from the optimisation objective.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_retriever_diagnostic(quality_df: "pd.DataFrame",
                                out_path: str | Path) -> dict:
    """Aggregate per-query context P/R; write a CSV; return summary stats."""
    if quality_df.empty:
        raise ValueError("quality_df is empty")

    spec = quality_df[quality_df["agent"] != "A_dispatcher"].copy()
    have = [c for c in ("q_context_precision", "q_context_recall") if c in spec.columns]
    if not have:
        return {"n_queries": 0, "context_precision": float("nan"),
                "context_recall": float("nan")}

    # All candidates retrieved the SAME contexts for a given query, so any
    # (query_id) row's P/R is the same across candidates; take one per query.
    per_q = (spec.groupby("query_id")[have].first()
                 .reset_index()
                 .rename(columns={c: c[2:] for c in have}))
    per_q.to_csv(out_path, index=False, float_format="%.4f")

    summary = {
        "n_queries": int(per_q.shape[0]),
        "context_precision": float(per_q["context_precision"].mean()) if "context_precision" in per_q.columns else float("nan"),
        "context_recall": float(per_q["context_recall"].mean()) if "context_recall" in per_q.columns else float("nan"),
    }
    return summary
