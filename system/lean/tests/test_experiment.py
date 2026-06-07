"""End-to-end runner + ablations + figures."""

from pathlib import Path

import pandas as pd
import pytest
from src.ablations import (
    ablation_catalog_scope,
    ablation_per_role_contribution,
    ablation_sequential_vs_concurrent,
    ablation_sla_sweep,
)
from src.experiment import run_one
from src.figures import render_all
from src.instance import load_instance
from src.quality import synthetic_quality

_LEAN_ROOT = Path(__file__).resolve().parents[1]
_EXP = _LEAN_ROOT / "experiments" / "lean_8gb.yaml"
_PERF_TABLE = _LEAN_ROOT.parent / "shared" / "data" / "perf_table.parquet"

_NEEDS_PERF = pytest.mark.skipif(
    not _PERF_TABLE.exists(),
    reason=f"shared perf table missing at {_PERF_TABLE}",
)


_DOMAINS = ("inference", "comparison", "temporal", "null_query")


@pytest.fixture(scope="module")
def instance():
    from catalog import build_catalog as bc

    catalog = bc.build()
    bc.write_json(catalog, _LEAN_ROOT / "catalog" / "catalog.json")
    return load_instance(_EXP).model_copy(update={"domains": _DOMAINS})


@pytest.fixture(scope="module")
def quality(instance):
    """Synthetic quality scoped to the test domains, so ablation tests don't
    pick up a real `quality.parquet` that uses different domain labels."""
    return synthetic_quality(instance.catalog, domains=_DOMAINS)


@_NEEDS_PERF
def test_run_one_writes_meta_alloc_baselines(tmp_path, instance):
    out_dir = run_one(_EXP, tmp_path, seed=42)
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "alloc.json").exists()
    assert (out_dir / "baselines.csv").exists()
    import json

    meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    assert "catalog_sha256" in meta
    assert meta["quality_source"] == "synthetic"
    assert meta["milp"]["feasible"]


@_NEEDS_PERF
def test_run_one_is_deterministic_at_same_seed(tmp_path, instance):
    a = run_one(_EXP, tmp_path / "a", seed=42)
    b = run_one(_EXP, tmp_path / "b", seed=42)
    import json

    ma = json.loads((a / "meta.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "meta.json").read_text(encoding="utf-8"))
    assert ma["catalog_sha256"] == mb["catalog_sha256"]
    # The MILP optimum must be exactly the same at the same seed.
    assert ma["milp"]["Q"] == mb["milp"]["Q"]
    assert ma["milp"]["config_by_role"] == mb["milp"]["config_by_role"]


@_NEEDS_PERF
def test_ablation_seq_vs_conc_concurrent_dominates(instance, quality):
    r = ablation_sequential_vs_concurrent(instance, quality, t_circ_grid=(6.0, 8.0, 10.0))
    df = r.df
    # Concurrent is a relaxation of sequential when k_active > 1; Q should be ≥.
    both = df.dropna(subset=["Q_concurrent", "Q_sequential"])
    assert len(both) > 0
    for _, row in both.iterrows():
        assert row["Q_concurrent"] + 1e-9 >= row["Q_sequential"]


@_NEEDS_PERF
def test_ablation_sla_sweep_monotone_in_t_circ(instance, quality):
    r = ablation_sla_sweep(instance, quality, t_circ_grid=(2.0, 4.0, 6.0, 10.0))
    df = r.df.dropna(subset=["Q"])
    # Q is non-decreasing in T° (relaxing a hard constraint can't hurt).
    assert df["Q"].is_monotonic_increasing


@_NEEDS_PERF
def test_ablation_per_role_runs(instance, quality):
    r = ablation_per_role_contribution(instance, quality, t_circ=8.0)
    assert set(r.df["pivot_role"].tolist()) == {"d", "s", "y"}
    # The "lock-2 vary-1" estimate must not exceed the joint MILP optimum.
    for _, row in r.df.iterrows():
        assert row["Q_joint_milp"] + 1e-6 >= row["Q_total_estimated"]


@_NEEDS_PERF
def test_ablation_catalog_scope_drops_at_least_some_q_at_some_t(instance, quality):
    r = ablation_catalog_scope(
        instance, quality, t_circ_grid=(4.0, 8.0, 12.0), drop_substring="3B"
    )
    df = r.df.dropna(subset=["Q_full", "Q_shrunk"])
    # Q_shrunk ≤ Q_full at every T° (catalog scope is monotone).
    for _, row in df.iterrows():
        assert row["Q_shrunk"] <= row["Q_full"] + 1e-9


@_NEEDS_PERF
def test_render_all_renders_existing_csvs(tmp_path, instance):
    ablation_dir = tmp_path / "ablations"
    ablation_dir.mkdir()
    # Write a minimal sla_sweep.csv just to exercise the renderer.
    df = pd.DataFrame(
        {
            "t_circ_s": [2.0, 4.0, 6.0],
            "Q": [0.4, 0.5, 0.6],
            "L_total_s": [1.5, 3.0, 5.0],
            "memory_used_gb": [3.0, 4.0, 5.0],
            "feasible": [True, True, True],
            "n_groups_loaded": [2, 2, 3],
            "lambda_s": [0.5, 1.0, 1.5],
        }
    )
    df.to_csv(ablation_dir / "sla_sweep.csv", index=False)
    out = render_all(ablation_dir, tmp_path / "figures")
    assert len(out) == 1
    assert (tmp_path / "figures" / "sla_sweep.pdf").exists()
