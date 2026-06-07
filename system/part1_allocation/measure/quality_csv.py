"""
part1_allocation/measure/quality_csv.py
=======================================
Export the per-query quality table to a CSV summary keyed by (agent, config_id).

One row per (agent, config_id), with the MEAN over queries of:
  faithfulness, answer_relevancy, context_precision, context_recall,
  correctness, quality (the aggregate), plus n (number of queries).

This is the headline artifact for a paper: a quality scorecard you can sort,
compare across models, and join to perf_table to plot quality-vs-cost.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

METRIC_COLS = [
    "q_faithfulness",
    "q_answer_relevancy",
    "q_context_precision",
    "q_context_recall",
    "q_correctness",
    "quality",
]


def export_quality_csv(quality_df: "pd.DataFrame", out_path: str | Path) -> "pd.DataFrame":
    """Aggregate the long quality table to one row per (agent, config_id) and
    write a CSV. Returns the aggregated dataframe."""
    if quality_df.empty:
        raise ValueError("quality_df is empty -- nothing to export")

    df = quality_df.copy()
    # Keep only metric cols actually present (router rows may not have all of them)
    cols = [c for c in METRIC_COLS if c in df.columns]

    # NaN-aware mean per (agent, config_id); separately count valid queries
    grouped = df.groupby(["agent", "config_id"], sort=True)
    agg = grouped[cols].mean(numeric_only=True).round(4)
    agg["n_queries"] = grouped.size()

    # Rename for a cleaner CSV (drop the q_ prefix; keep `quality` as is)
    rename = {c: c[2:] if c.startswith("q_") else c for c in cols}
    agg = agg.rename(columns=rename).reset_index()

    # Column order: identifiers, n, then metrics in a consistent order
    metric_order = ["faithfulness", "answer_relevancy", "context_precision",
                    "context_recall", "correctness", "quality"]
    metric_order = [m for m in metric_order if m in agg.columns]
    agg = agg[["agent", "config_id", "n_queries"] + metric_order]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False, float_format="%.4f")
    return agg
