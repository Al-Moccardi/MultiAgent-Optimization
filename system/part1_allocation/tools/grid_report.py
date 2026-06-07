"""
part1_allocation/tools/grid_report.py
======================================
Generate two PNG grids of mini-heatmaps for the paper:

  Figure 1 -- SPECIALISTS (2x5 mini-heatmaps over model x quant):
      Row 1: faithfulness | answer_relevancy | context_precision | context_recall | correctness
      Row 2: quality (aggregate) | peak memory | TTFT | throughput | energy
      Aggregated across context length AND across all specialist agents.

  Figure 2 -- DISPATCHER (1x5 mini-heatmaps):
      Routing F1 | peak memory | TTFT | throughput | energy

Each cell shows its numeric value; missing (model, quant) combinations (e.g.
Q3_K_M unavailable for some models) are shown blank/grey. Lower-is-better
metrics (mem/TTFT/energy) use a reversed colormap.

Inputs (defaults):
  shared/pareto/quality_scorecard.csv   (written by run_all)
  shared/pareto/perf_table.parquet      (written by run_all)

Usage:
  python -m part1_allocation.tools.grid_report
  python -m part1_allocation.tools.grid_report --run-dir custom/out --out-dir paper/figs
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def split_config(cid: str) -> tuple[str, str, str]:
    """qwen2.5-3b__Q4_K_M__c4096 -> ('qwen2.5-3b', 'Q4_K_M', 'c4096')."""
    parts = cid.split("__")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return cid, "", ""


def _model_size(m: str) -> float:
    """Best-effort param count from the model key, for consistent y-axis ordering."""
    s = m.lower()
    if "360m" in s:
        return 0.36
    m1 = re.search(r"(\d+(?:[._]\d+)?)b", s)
    if m1:
        return float(m1.group(1).replace("_", "."))
    return 99.0


def _quant_order(q: str) -> int:
    """Q3 < Q4 < Q5 < Q6 < Q8 (numeric bit-width)."""
    m = re.search(r"Q(\d)", q)
    return int(m.group(1)) if m else 99


def _heatmap(ax, piv: pd.DataFrame, title: str, *, fmt: str = ".2f",
             cmap: str = "viridis", vmin=None, vmax=None,
             lower_is_better: bool = False) -> None:
    if lower_is_better:
        cmap = cmap + "_r" if not cmap.endswith("_r") else cmap[:-2]
    arr = piv.values.astype(float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.tick_params(axis="both", length=0)

    # numeric labels with auto-contrast text colour
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#888")
            else:
                txt_color = "white" if im.norm(v) > 0.55 else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        fontsize=7, color=txt_color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)


def _pivot(df: pd.DataFrame, value_col: str,
           model_order: list[str], quant_order: list[str]) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(index=model_order, columns=quant_order, dtype=float)
    agg = df.groupby(["model", "quant"], as_index=False)[value_col].mean()
    return (agg.pivot(index="model", columns="quant", values=value_col)
              .reindex(index=model_order, columns=quant_order))


# Each entry: (column, title, vmin, vmax, base_cmap, lower_is_better, fmt)
_SPEC_METRICS = [
    ("faithfulness",       "Faithfulness",          0, 1, "viridis", False, ".2f"),
    ("answer_relevancy",   "Answer relevancy",      0, 1, "viridis", False, ".2f"),
    ("context_precision",  "Context precision",     0, 1, "viridis", False, ".2f"),
    ("context_recall",     "Context recall",        0, 1, "viridis", False, ".2f"),
    ("correctness",        "Correctness",           0, 1, "viridis", False, ".2f"),
    ("quality",            "Quality (aggregate)",   0, 1, "magma",   False, ".2f"),
    ("peak_mem_gb",        "Peak memory (GB)",   None, None, "magma",   True,  ".2f"),
    ("ttft_s",             "TTFT (s)",           None, None, "magma",   True,  ".2f"),
    ("throughput_tok_s",   "Throughput (tok/s)", None, None, "viridis", False, ".1f"),
    ("energy_j_per_tok",   "Energy (J/tok)",     None, None, "magma",   True,  ".2f"),
]

_DISP_METRICS = [
    ("quality",            "Routing F1",          0, 1, "viridis", False, ".3f"),
    ("peak_mem_gb",        "Peak memory (GB)", None, None, "magma",   True,  ".2f"),
    ("ttft_s",             "TTFT (s)",         None, None, "magma",   True,  ".2f"),
    ("throughput_tok_s",   "Throughput (tok/s)", None, None, "viridis", False, ".1f"),
    ("energy_j_per_tok",   "Energy (J/tok)",   None, None, "magma",   True,  ".2f"),
]

# Context figure: rows are (model, quant) pairs, columns are contexts.
# Cells are RATIOS to the smallest measured context per (m,q), so you read the
# MARGINAL cost of growing n_ctx. Quality also shown to expose context-fit drops.
_CTX_METRICS = [
    ("peak_mem_gb",       "Peak memory (× c-min)",   "magma",   True,  ".2f"),
    ("ttft_s",            "TTFT (× c-min)",          "magma",   True,  ".2f"),
    ("throughput_tok_s",  "Throughput (× c-min)",    "viridis", False, ".2f"),
    ("energy_j_per_tok",  "Energy (× c-min)",        "magma",   True,  ".2f"),
    ("quality",           "Specialist quality (abs)", "viridis", False, ".2f"),
]


def _ctx_order(c: str) -> int:
    """c2048 < c4096 < c8192 ... by the numeric part."""
    m = re.search(r"\d+", c or "")
    return int(m.group()) if m else 99


def _ratio_to_min(df_long: pd.DataFrame, value_col: str,
                  row_order: list[tuple[str, str]],
                  ctx_order: list[str]) -> pd.DataFrame:
    """Pivot to (model, quant) x context and divide each row by its leftmost
    (smallest-context) non-NaN value, so cells are unitless ratios. The base
    column is always 1.00 by construction."""
    df_long = df_long.copy()
    df_long["mq"] = list(zip(df_long["model"], df_long["quant"]))
    piv = (df_long.groupby(["mq", "ctx"], as_index=False)[value_col].mean()
                  .pivot(index="mq", columns="ctx", values=value_col))
    piv = piv.reindex(index=row_order, columns=ctx_order)
    base = piv.bfill(axis=1).iloc[:, 0]   # smallest context per row
    return piv.div(base, axis=0)


def _quality_abs_pivot(df_long: pd.DataFrame,
                       row_order: list[tuple[str, str]],
                       ctx_order: list[str]) -> pd.DataFrame:
    """Quality on its absolute 0-1 scale; lets the eye spot context-fit drops
    where the small-context variant could not fit a long RAG prompt."""
    df_long = df_long.copy()
    df_long["mq"] = list(zip(df_long["model"], df_long["quant"]))
    piv = (df_long.groupby(["mq", "ctx"], as_index=False)["quality"].mean()
                  .pivot(index="mq", columns="ctx", values="quality"))
    return piv.reindex(index=row_order, columns=ctx_order)


def _make_context_figure(spec_panel_df: pd.DataFrame, out_path: Path,
                         n_ctxs: int, n_pairs: int,
                         group_by_model: bool = False) -> Path:
    if group_by_model:
        # Compact: one row per model (average across quants), good for main paper.
        df = spec_panel_df.copy()
        df["quant"] = "(avg)"
        models = sorted(set(df["model"]), key=_model_size)
        pairs = [(m, "(avg)") for m in models]
        row_labels = models
        title_suffix = "averaged across quants — one row per model"
    else:
        # Full: every (model, quant) row.
        pairs = sorted({(m, q) for m, q in zip(spec_panel_df["model"], spec_panel_df["quant"])},
                       key=lambda mq: (_model_size(mq[0]), _quant_order(mq[1])))
        df = spec_panel_df
        row_labels = [f"{m} · {q}" for (m, q) in pairs]
        title_suffix = f"{len(pairs)} (model · quantization) rows"

    ctxs = sorted(set(df["ctx"]), key=_ctx_order)

    height = max(4, 0.32 * len(pairs) + 2)
    fig, axes = plt.subplots(1, 5, figsize=(22, height), constrained_layout=False)
    for ax, (col, title, cmap, lower, fmt) in zip(axes, _CTX_METRICS):
        if col == "quality":
            piv = _quality_abs_pivot(df, pairs, ctxs)
            vmin, vmax = 0.0, 1.0
        else:
            piv = _ratio_to_min(df, col, pairs, ctxs)
            vmin, vmax = None, None
        piv.index = row_labels
        _heatmap(ax, piv, title, vmin=vmin, vmax=vmax, cmap=cmap,
                 lower_is_better=lower, fmt=fmt)

    fig.suptitle(
        f"Context-window cost grid — context length on x-axis ({title_suffix}; "
        f"performance shown as RATIO vs each row's smallest measured context; "
        f"quality on its absolute 0-1 scale)",
        fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def make_figures(scorecard_path: Path, perf_path: Path, out_dir: Path) -> dict:
    score = pd.read_csv(scorecard_path)
    perf = pd.read_parquet(perf_path)

    for df in (score, perf):
        parts = df["config_id"].apply(split_config).tolist()
        df["model"] = [p[0] for p in parts]
        df["quant"] = [p[1] for p in parts]
        df["ctx"] = [p[2] for p in parts]

    model_order = sorted(set(score["model"]) | set(perf["model"]), key=_model_size)
    quant_order = sorted(set(score["quant"]) | set(perf["quant"]), key=_quant_order)

    # -------- Figure 1: specialists -----------------------------------------
    spec = score[score["agent"] != "A_dispatcher"].copy()
    # Specialist quality is summarised across context AND across agents.
    spec_panel_df = spec.merge(perf, on=["config_id", "model", "quant", "ctx"], how="left")

    fig1, axes1 = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=False)
    for ax, (col, title, vmin, vmax, cmap, lower, fmt) in zip(axes1.flat, _SPEC_METRICS):
        piv = _pivot(spec_panel_df, col, model_order, quant_order)
        _heatmap(ax, piv, title, vmin=vmin, vmax=vmax, cmap=cmap,
                 lower_is_better=lower, fmt=fmt)
    n_cells = sum(1 for _ in spec.groupby(["model", "quant"]))
    fig1.suptitle(
        f"Specialist evaluation grid — model × quantization "
        f"(averaged across {spec['ctx'].nunique()} contexts and "
        f"{spec['agent'].nunique()} specialist agents; {n_cells} cells)",
        fontsize=13, fontweight="bold", y=0.995)
    fig1.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = out_dir / "grid_specialists.png"
    fig1.savefig(out1, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    # -------- Figure 2: dispatcher -----------------------------------------
    disp = score[score["agent"] == "A_dispatcher"].copy()
    disp_panel_df = disp.merge(perf, on=["config_id", "model", "quant", "ctx"], how="left")

    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 5), constrained_layout=False)
    for ax, (col, title, vmin, vmax, cmap, lower, fmt) in zip(axes2, _DISP_METRICS):
        piv = _pivot(disp_panel_df, col, model_order, quant_order)
        _heatmap(ax, piv, title, vmin=vmin, vmax=vmax, cmap=cmap,
                 lower_is_better=lower, fmt=fmt)
    fig2.suptitle(
        f"Dispatcher (routing) grid — model × quantization "
        f"(averaged across {disp['ctx'].nunique()} contexts; routing F1 + performance)",
        fontsize=13, fontweight="bold", y=0.995)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = out_dir / "grid_dispatcher.png"
    fig2.savefig(out2, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig2)

    # -------- Figure 3: context-window cost (full + compact) -------------
    out3 = _make_context_figure(spec_panel_df, out_dir / "grid_context.png",
                                n_ctxs=spec_panel_df["ctx"].nunique(),
                                n_pairs=spec_panel_df.groupby(["model","quant"]).ngroups,
                                group_by_model=False)
    out3c = _make_context_figure(spec_panel_df, out_dir / "grid_context_compact.png",
                                 n_ctxs=spec_panel_df["ctx"].nunique(),
                                 n_pairs=spec_panel_df["model"].nunique(),
                                 group_by_model=True)

    return {"specialists": out1, "dispatcher": out2,
            "context": out3, "context_compact": out3c,
            "n_models": len(model_order), "n_quants": len(quant_order),
            "n_spec_cells": n_cells}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the experimental grid PNGs")
    ap.add_argument("--run-dir", default="shared/pareto",
                    help="directory containing quality_scorecard.csv and perf_table.parquet")
    ap.add_argument("--out-dir", default="shared/figures",
                    help="directory to write the PNGs")
    args = ap.parse_args(argv)

    rd = Path(args.run_dir)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    sc = rd / "quality_scorecard.csv"
    pf = rd / "perf_table.parquet"
    if not sc.exists():
        raise SystemExit(f"missing {sc}; run the pipeline first")
    if not pf.exists():
        raise SystemExit(f"missing {pf}; run the pipeline first")

    info = make_figures(sc, pf, od)
    print(f"[grid] models={info['n_models']}  quants={info['n_quants']}  "
          f"specialist cells={info['n_spec_cells']}")
    print(f"[grid] -> {info['specialists']}")
    print(f"[grid] -> {info['dispatcher']}")
    print(f"[grid] -> {info['context']}  (full, appendix)")
    print(f"[grid] -> {info['context_compact']}  (compact, main paper)")


if __name__ == "__main__":
    main()
