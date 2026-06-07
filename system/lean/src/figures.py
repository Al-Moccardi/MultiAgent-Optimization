"""Render PDFs from the ablation CSVs (paper-grade matplotlib)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_PAPER_RC = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
}


def apply_style() -> None:
    mpl.rcParams.update(_PAPER_RC)


def plot_sequential_vs_concurrent(csv: Path, out: Path) -> None:
    apply_style()
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df["t_circ_s"], df["Q_concurrent"], "o-", color="#1f77b4", label="concurrent")
    ax.plot(df["t_circ_s"], df["Q_sequential"], "s--", color="#d62728", label="sequential (k=3)")
    ax.set_xlabel("$T^\\circ$ (s)")
    ax.set_ylabel("optimal $Q$")
    ax.set_title("Sequential vs concurrent latency")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_sla_sweep(csv: Path, out: Path) -> None:
    apply_style()
    df = pd.read_csv(csv)
    feasible = df[df["feasible"].astype(bool)]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(feasible["t_circ_s"], feasible["Q"], "o-", color="#1f77b4", label="MILP $Q^\\star$")
    ax2 = ax.twinx()
    ax2.plot(feasible["t_circ_s"], feasible["memory_used_gb"], "x:", color="#7f7f7f", label="memory used (GB)")
    ax.set_xlabel("$T^\\circ$ (s)")
    ax.set_ylabel("optimal $Q$")
    ax2.set_ylabel("memory used (GB)")
    ax.set_title("SLA sweep — $Q^\\star(T^\\circ)$")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_per_role_contribution(csv: Path, out: Path) -> None:
    apply_style()
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = range(len(df))
    ax.bar(
        [i - 0.18 for i in x],
        df["Q_total_estimated"],
        width=0.36,
        label="lock 2 / vary 1 (best alone)",
        color="#1f77b4",
    )
    ax.bar(
        [i + 0.18 for i in x],
        df["Q_joint_milp"],
        width=0.36,
        label="joint MILP",
        color="#2ca02c",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["pivot_role"])
    ax.set_xlabel("pivot role (others locked to min-mem)")
    ax.set_ylabel("$Q$")
    ax.set_title("Per-role quality contribution")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_catalog_scope(csv: Path, out: Path) -> None:
    apply_style()
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df["t_circ_s"], df["Q_full"], "o-", color="#1f77b4", label="full catalog")
    ax.plot(df["t_circ_s"], df["Q_shrunk"], "s--", color="#d62728", label="3B dropped")
    ax.set_xlabel("$T^\\circ$ (s)")
    ax.set_ylabel("optimal $Q$")
    ax.set_title("Catalog scope — marginal value of 3B")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def render_all(ablation_dir: Path, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, render in [
        ("sequential_vs_concurrent", plot_sequential_vs_concurrent),
        ("sla_sweep", plot_sla_sweep),
        ("per_role_contribution", plot_per_role_contribution),
        ("catalog_scope", plot_catalog_scope),
    ]:
        csv = ablation_dir / f"{name}.csv"
        if not csv.exists():
            continue
        out = figures_dir / f"{name}.pdf"
        render(csv, out)
        paths.append(out)
    return paths


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--ablations",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ablations",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "figures",
    )
    args = p.parse_args(argv)
    paths = render_all(args.ablations, args.out)
    for p_ in paths:
        print(f"[figures] wrote {p_}")


if __name__ == "__main__":
    main()
