"""The four lean ablations (paper §results).

1. **Sequential vs concurrent latency** at multiple $T^\\circ$ points.
2. **SLA tightness sweep** $T^\\circ \\in [\\text{low}, \\text{high}]$ s.
3. **Per-role quality contribution** — lock two roles to their min-mem
   feasible config, vary the third.
4. **Catalog scope** — drop the 3B model and re-solve at each $T^\\circ$.

A fifth (sanity-vs-colleague) is left for the whitepaper text; the colleague's
parts/1–3 use a different dataset (Italian family-law) and re-running their
allocations on a foreign metric set would be misleading.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.instance import load_instance
from src.milp import solve_milp
from src.quality import Quality, synthetic_quality
from src.types import Catalog, Instance, Role

_DEFAULT_QUALITY_PARQUET = Path(__file__).resolve().parents[1] / "catalog" / "quality.parquet"


@dataclass(frozen=True)
class AblationResult:
    name: str
    df: pd.DataFrame
    notes: dict[str, str]


def _quality_for(
    instance: Instance, quality_path: Path | None = None
) -> tuple[Quality, str]:
    """Real `quality.parquet` if available; synthetic_quality otherwise.

    Returns ``(quality, source_tag)`` so callers can record provenance.
    """
    if quality_path is None and _DEFAULT_QUALITY_PARQUET.exists():
        quality_path = _DEFAULT_QUALITY_PARQUET
    if quality_path is not None and quality_path.exists():
        return Quality.from_parquet(quality_path), f"file:{quality_path.name}"
    domains = instance.domains or ("default",)
    return synthetic_quality(instance.catalog, domains=domains), "synthetic"


# ---------------------------------------------------------------------------
# 1. Sequential vs concurrent latency
# ---------------------------------------------------------------------------


def ablation_sequential_vs_concurrent(
    instance: Instance,
    quality: Quality,
    t_circ_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0),
    k_active_seq: int = 3,
) -> AblationResult:
    """For each $T^\\circ$, solve once concurrent and once sequential (k_active).

    Reports the Q gap (relaxation gain) and the L_total achieved by each model.
    """
    rows: list[dict] = []
    for t in t_circ_grid:
        inst = instance.model_copy(update={"t_circ_s": float(t)})
        try:
            conc = solve_milp(inst, quality, latency_model="concurrent")
        except ValueError:
            conc = None
        try:
            seq = solve_milp(inst, quality, latency_model="sequential", k_active=k_active_seq)
        except ValueError:
            seq = None
        rows.append(
            {
                "t_circ_s": float(t),
                "Q_concurrent": conc.allocation.Q if conc and conc.allocation else None,
                "Q_sequential": seq.allocation.Q if seq and seq.allocation else None,
                "L_concurrent": conc.allocation.L_total_s if conc and conc.allocation else None,
                "L_sequential": seq.allocation.L_total_s if seq and seq.allocation else None,
                "concurrent_feasible": bool(conc and conc.allocation),
                "sequential_feasible": bool(seq and seq.allocation),
            }
        )
    df = pd.DataFrame(rows)
    return AblationResult(
        name="sequential_vs_concurrent",
        df=df,
        notes={
            "k_active": str(k_active_seq),
            "interpretation": (
                "Concurrent should yield Q ≥ sequential at every T° "
                "(strictly larger feasible region)."
            ),
        },
    )


# ---------------------------------------------------------------------------
# 2. SLA tightness sweep
# ---------------------------------------------------------------------------


def ablation_sla_sweep(
    instance: Instance,
    quality: Quality,
    t_circ_grid: tuple[float, ...] = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0),
) -> AblationResult:
    """Q^\\star(T°) — the headline Pareto curve."""
    rows: list[dict] = []
    for t in t_circ_grid:
        inst = instance.model_copy(update={"t_circ_s": float(t)})
        try:
            r = solve_milp(inst, quality)
        except ValueError as exc:
            rows.append({"t_circ_s": float(t), "Q": None, "L_total_s": None, "feasible": False, "error": str(exc)})
            continue
        if r.allocation is None:
            rows.append({"t_circ_s": float(t), "Q": None, "L_total_s": None, "feasible": False})
            continue
        a = r.allocation
        rows.append(
            {
                "t_circ_s": float(t),
                "Q": a.Q,
                "L_total_s": a.L_total_s,
                "memory_used_gb": a.memory_used_gb,
                "n_groups_loaded": len(a.loaded_groups),
                "feasible": a.feasible,
                "lambda_s": r.diagnostics.get("lambda"),
            }
        )
    return AblationResult(
        name="sla_sweep",
        df=pd.DataFrame(rows),
        notes={"interpretation": "Q is non-decreasing in T°; the elbow signals the binding SLA."},
    )


# ---------------------------------------------------------------------------
# 3. Per-role quality contribution
# ---------------------------------------------------------------------------


def ablation_per_role_contribution(
    instance: Instance,
    quality: Quality,
    t_circ: float = 8.0,
) -> AblationResult:
    """Quantify which role drives Q.

    For each role $r^\\star$: lock the other two roles to their min-`weight_gb`
    config (eligible under the SLA), then maximise Q over $r^\\star$ alone by
    iterating its eligible configs. Report the best Q and how it compares to
    the joint MILP optimum.
    """
    from src.instance import build_arrays

    inst = instance.model_copy(update={"t_circ_s": float(t_circ)})
    arr = build_arrays(inst)

    # Joint MILP optimum (reference).
    joint = solve_milp(inst, quality).allocation
    if joint is None:
        raise RuntimeError("joint MILP infeasible at t_circ for ablation 3")

    # Use coverage-filtered eligibility so unmeasured configs aren't picked.
    from src.quality_coverage import covered_eligibility, per_config_coefficients

    q_coefs = per_config_coefficients(arr, quality, inst.domains or ("default",))
    covered = covered_eligibility(arr, q_coefs)

    # Re-pick cheapest among covered configs (not SLA-only).
    def _cheapest_covered_for(role: Role) -> int:
        elig = covered[role].tolist()
        best_k = elig[0]
        best_key = (arr.weights_g[int(arr.group_of_k[best_k])], arr.kv_gb[best_k])
        for k in elig[1:]:
            key = (arr.weights_g[int(arr.group_of_k[k])], arr.kv_gb[k])
            if key < best_key:
                best_key = key
                best_k = k
        return int(best_k)

    cheapest = {r: _cheapest_covered_for(r) for r in Role}

    rows: list[dict] = []
    for pivot in Role:
        Q_locked = sum(
            float(q_coefs[r][cheapest[r]]) for r in Role if r is not pivot
        )
        # Best pivot value over coverage-filtered eligibility.
        elig = covered[pivot].tolist()
        best_q = max(float(q_coefs[pivot][k]) for k in elig)
        rows.append(
            {
                "pivot_role": pivot.value,
                "Q_locked_others": Q_locked,
                "Q_pivot_best_alone": best_q,
                "Q_total_estimated": Q_locked + best_q,
                "Q_joint_milp": joint.Q,
                "delta_vs_joint": joint.Q - (Q_locked + best_q),
            }
        )
    return AblationResult(
        name="per_role_contribution",
        df=pd.DataFrame(rows),
        notes={
            "t_circ": str(t_circ),
            "interpretation": (
                "Per-role-best ignores interaction; the joint MILP can exceed "
                "the simple sum when sharing a group frees memory for a richer "
                "config elsewhere."
            ),
        },
    )


# ---------------------------------------------------------------------------
# 4. Catalog scope (drop 3B)
# ---------------------------------------------------------------------------


def ablation_catalog_scope(
    instance: Instance,
    quality: Quality,
    t_circ_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0),
    drop_substring: str = "3B",
) -> AblationResult:
    """Solve with the full catalog vs a shrunk catalog (drop configs whose
    `model_id` contains `drop_substring`).

    The Q gap at each $T^\\circ$ is the marginal value of the dropped tier. The
    same `quality` is used for both the full and the shrunk problem — the
    shrunk problem's `Quality` is the natural restriction (only configs that
    remain in the catalog have entries that matter).
    """
    full_quality = quality
    shrunk_catalog = Catalog(
        configs=tuple(
            c for c in instance.catalog.configs if drop_substring not in c.model_id
        )
    )
    shrunk_instance = instance.model_copy(update={"catalog": shrunk_catalog})
    shrunk_quality = quality

    rows: list[dict] = []
    for t in t_circ_grid:
        full = instance.model_copy(update={"t_circ_s": float(t)})
        shrunk = shrunk_instance.model_copy(update={"t_circ_s": float(t)})
        try:
            r_full = solve_milp(full, full_quality).allocation
        except ValueError:
            r_full = None
        try:
            r_shr = solve_milp(shrunk, shrunk_quality).allocation
        except ValueError:
            r_shr = None
        rows.append(
            {
                "t_circ_s": float(t),
                "Q_full": r_full.Q if r_full else None,
                "Q_shrunk": r_shr.Q if r_shr else None,
                "delta_Q": (r_full.Q - r_shr.Q) if (r_full and r_shr) else None,
            }
        )
    return AblationResult(
        name="catalog_scope",
        df=pd.DataFrame(rows),
        notes={
            "dropped": drop_substring,
            "interpretation": "delta_Q quantifies the marginal value of the dropped model tier.",
        },
    )


# ---------------------------------------------------------------------------
# Run all four — used by the CLI.
# ---------------------------------------------------------------------------


def run_all(instance: Instance, quality: Quality) -> dict[str, AblationResult]:
    return {
        "sequential_vs_concurrent": ablation_sequential_vs_concurrent(instance, quality),
        "sla_sweep": ablation_sla_sweep(instance, quality),
        "per_role_contribution": ablation_per_role_contribution(instance, quality),
        "catalog_scope": ablation_catalog_scope(instance, quality),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("yaml", type=Path, help="Experiment YAML")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "ablations",
    )
    p.add_argument(
        "--quality",
        type=Path,
        default=None,
        help="Path to quality.parquet (defaults to catalog/quality.parquet if present, else synthetic)",
    )
    args = p.parse_args(argv)
    instance = load_instance(args.yaml)
    quality, source = _quality_for(instance, args.quality)
    print(f"[ablations] quality source: {source}")
    results = run_all(instance, quality)
    args.out.mkdir(parents=True, exist_ok=True)
    for r in results.values():
        r.df.to_csv(args.out / f"{r.name}.csv", index=False)
        (args.out / f"{r.name}.notes.txt").write_text(
            "\n".join(f"{k}: {v}" for k, v in r.notes.items()), encoding="utf-8"
        )
        print(f"[ablations] wrote {r.name}.csv ({len(r.df)} rows)")


if __name__ == "__main__":
    main()
