"""
part1_allocation/tools/bootstrap_report.py
===========================================
CLI for the optimum-robustness bootstrap (Tier-3, Finding 8).

Runs the query-bootstrap on a SAVED bundle (the parquets the pipeline already
wrote) and prints objective confidence intervals + per-agent selection
stability. No model inference -- pure resampling of the measured quality table.

Example:
    python -m part1_allocation.tools.bootstrap_report \
        --bundle shared/pareto \
        --agents part1_allocation/config/agents.yaml \
        --catalog part1_allocation/config/catalog.yaml \
        --device part1_allocation/config/device.yaml \
        --n-boot 300 --latency-model worst_case

The bundle dir must contain quality_table.parquet, perf_table.parquet, and
manifest.json (written by run_all). The hardware string is read from the
perf table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from part1_allocation.config_loader import load_agents, load_catalog, load_device
from part1_allocation.optimize.bootstrap import bootstrap_optimum, print_bootstrap_summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bootstrap confidence for the MAMAP optimum")
    ap.add_argument("--bundle", default="shared/pareto",
                    help="dir with quality_table.parquet + perf_table.parquet")
    ap.add_argument("--agents", default="part1_allocation/config/agents.yaml")
    ap.add_argument("--catalog", default="part1_allocation/config/catalog.yaml")
    ap.add_argument("--device", default="part1_allocation/config/device.yaml")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--eps", type=float, default=None,
                    help="latency cap for each resample's solve (default: unconstrained)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latency-model", choices=["worst_case", "expected_max"],
                    default="worst_case")
    ap.add_argument("--normalize-specialists", action="store_true")
    ap.add_argument("--obj-weights", default=None, help="'w_rt,w_syn,w_spec'")
    ap.add_argument("--out", default=None, help="optional path to write the summary JSON")
    args = ap.parse_args(argv)

    b = Path(args.bundle)
    quality_df = pd.read_parquet(b / "quality_table.parquet")
    perf_df = pd.read_parquet(b / "perf_table.parquet")
    hardware = perf_df["hardware"].iloc[0]

    agents = load_agents(args.agents)
    configs = load_catalog(args.catalog)
    device = load_device(args.device)

    solve_kwargs = {"latency_model": args.latency_model,
                    "normalize_specialists": args.normalize_specialists}
    if args.obj_weights:
        parts = [float(x) for x in args.obj_weights.split(",")]
        if len(parts) != 3:
            raise SystemExit("--obj-weights must be 'w_rt,w_syn,w_spec'")
        solve_kwargs["weights"] = tuple(parts)

    print(f"[bootstrap] hardware={hardware}  n_boot={args.n_boot}  "
          f"eps={args.eps}  latency_model={args.latency_model}")
    summary = bootstrap_optimum(quality_df, perf_df, agents, configs, device, hardware,
                                n_boot=args.n_boot, eps=args.eps, seed=args.seed,
                                solve_kwargs=solve_kwargs)
    print_bootstrap_summary(summary)

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"[bootstrap] summary -> {args.out}")


if __name__ == "__main__":
    main()
