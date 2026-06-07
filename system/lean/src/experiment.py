"""Experiment runner: solve MILP + 4 baselines, write a self-describing meta.json.

Usage (single experiment):
    python -m src.experiment experiments/lean_8gb.yaml

Each invocation writes a fresh `results/<run_id>/` with:
- `meta.json`     — catalog hash, master seed, lib versions, instance hyperparams
- `alloc.json`    — the MILP-optimal Allocation
- `baselines.csv` — one row per baseline (Q, L_total, memory_used, feasible)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import (
    solve_largest_fits,
    solve_per_role_best,
    solve_random_feasible,
    solve_uniform,
)
from src.instance import load_instance
from src.milp import solve_milp
from src.quality import Quality, synthetic_quality
from src.types import Allocation, Instance


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "unknown"


def _catalog_sha256(instance: Instance) -> str:
    payload = json.dumps(
        [
            {
                "model_id": c.group.model_id,
                "quant": c.group.quant,
                "context_length": c.context_length,
                "ttft_s": c.ttft_s,
                "throughput_tps": c.throughput_tps,
                "energy_j_per_tok": c.energy_j_per_tok,
            }
            for c in instance.catalog.configs
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _alloc_to_dict(a: Allocation) -> dict[str, Any]:
    return {
        "instance_name": a.instance_name,
        "Q": a.Q,
        "L_total_s": a.L_total_s,
        "memory_used_gb": a.memory_used_gb,
        "feasible": a.feasible,
        "source": a.source,
        "config_by_role": {r.value: cid for r, cid in a.config_by_role.items()},
        "loaded_groups": [list(g) for g in a.loaded_groups],
    }


def load_quality_for(
    instance: Instance, quality_path: Path | None = None
) -> tuple[Quality, str]:
    """Load `quality.parquet` if it exists; otherwise return synthetic scores
    and tag the source so meta.json declares it."""
    if quality_path is not None and quality_path.exists():
        return Quality.from_parquet(quality_path), f"file:{quality_path.name}"
    if not instance.domains:
        instance = instance.model_copy(
            update={"domains": ("default",)}
        )
    domains = instance.domains or ("default",)
    return synthetic_quality(instance.catalog, domains=domains), "synthetic"


def run_one(
    yaml_path: Path,
    results_root: Path,
    seed: int = 42,
    quality_path: Path | None = None,
    run_random: bool = True,
) -> Path:
    instance = load_instance(yaml_path)
    quality, quality_source = load_quality_for(instance, quality_path)

    run_id = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = results_root / instance.name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    milp = solve_milp(instance, quality)
    timings["milp"] = time.perf_counter() - t0
    if milp.allocation is None:
        raise RuntimeError("MILP infeasible — relax T_circ or M in the YAML")

    baselines: list[Allocation] = []
    for name, fn in [
        ("largest_fits", solve_largest_fits),
        ("per_role_best", solve_per_role_best),
        ("uniform", solve_uniform),
    ]:
        t0 = time.perf_counter()
        alloc = fn(instance, quality)
        timings[name] = time.perf_counter() - t0
        if alloc is not None:
            baselines.append(alloc)
    if run_random:
        t0 = time.perf_counter()
        alloc = solve_random_feasible(instance, quality, n_samples=500, seed=seed)
        timings["random_feasible"] = time.perf_counter() - t0
        if alloc is not None:
            baselines.append(alloc)

    (out_dir / "alloc.json").write_text(
        json.dumps(_alloc_to_dict(milp.allocation), indent=2), encoding="utf-8"
    )
    pd.DataFrame([_alloc_to_dict(a) for a in baselines]).to_csv(
        out_dir / "baselines.csv", index=False
    )

    meta = {
        "instance": instance.name,
        "yaml": str(yaml_path),
        "run_id": run_id,
        "catalog_sha256": _catalog_sha256(instance),
        "memory_gb": instance.memory_gb,
        "t_circ_s": instance.t_circ_s,
        "n_d": instance.n_d,
        "n_s": instance.n_s,
        "n_y": instance.n_y,
        "domains": list(instance.domains),
        "master_seed": seed,
        "quality_source": quality_source,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pkg_versions": {
            name: _pkg_version(name)
            for name in ["highspy", "pydantic", "numpy", "pandas"]
        },
        "timings_s": timings,
        "milp": _alloc_to_dict(milp.allocation),
        "milp_diagnostics": milp.diagnostics,
        "baselines": [_alloc_to_dict(a) for a in baselines],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("yaml", type=Path, help="Experiment YAML")
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="Optional quality.parquet (defaults to synthetic scores)",
    )
    args = p.parse_args(argv)
    out = run_one(args.yaml, args.results_root, seed=args.seed, quality_path=args.quality)
    print(f"[experiment] wrote {out}")


if __name__ == "__main__":
    main()
