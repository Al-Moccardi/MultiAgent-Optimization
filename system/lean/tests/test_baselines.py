"""The four naive baselines + MILP-beats-baselines sanity."""

from pathlib import Path

import pytest
from src.baselines import (
    solve_largest_fits,
    solve_per_role_best,
    solve_random_feasible,
    solve_uniform,
)
from src.instance import load_instance
from src.milp import solve_milp
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
def fixture():
    from catalog import build_catalog as bc

    catalog = bc.build()
    bc.write_json(catalog, _LEAN_ROOT / "catalog" / "catalog.json")
    instance = load_instance(_EXP).model_copy(update={"domains": _DOMAINS})
    quality = synthetic_quality(instance.catalog, domains=_DOMAINS)
    return instance, quality


@_NEEDS_PERF
def test_largest_fits_returns_feasible_or_none(fixture):
    instance, quality = fixture
    a = solve_largest_fits(instance, quality)
    if a is not None:
        assert a.feasible


@_NEEDS_PERF
def test_largest_fits_uses_one_loaded_group(fixture):
    instance, quality = fixture
    a = solve_largest_fits(instance, quality)
    if a is None:
        pytest.skip("no shareable config fits both T_circ and M")
    assert len(a.loaded_groups) == 1


@_NEEDS_PERF
def test_per_role_best_always_returns(fixture):
    instance, quality = fixture
    a = solve_per_role_best(instance, quality)
    assert a is not None
    # Feasibility not guaranteed — only that the allocation is constructed.


@_NEEDS_PERF
def test_uniform_returns_one_group_when_feasible(fixture):
    instance, quality = fixture
    a = solve_uniform(instance, quality)
    if a is not None:
        assert len(a.loaded_groups) == 1
        assert a.feasible


@_NEEDS_PERF
def test_random_feasible_returns_feasible(fixture):
    instance, quality = fixture
    a = solve_random_feasible(instance, quality, n_samples=50, seed=42)
    if a is not None:
        assert a.feasible
        assert a.memory_used_gb <= instance.memory_gb + 1e-6
        assert a.L_total_s <= instance.t_circ_s + 1e-6


@_NEEDS_PERF
def test_milp_beats_or_matches_all_baselines(fixture):
    """The MILP optimum is an upper bound — baselines can never strictly beat it."""
    instance, quality = fixture
    milp = solve_milp(instance, quality).allocation
    assert milp is not None

    baselines: list = []
    for fn in (solve_largest_fits, solve_uniform):
        a = fn(instance, quality)
        if a is not None and a.feasible:
            baselines.append(a)
    a = solve_per_role_best(instance, quality)
    if a is not None and a.feasible:
        baselines.append(a)
    a = solve_random_feasible(instance, quality, n_samples=100, seed=42)
    if a is not None:
        baselines.append(a)

    assert baselines, "no feasible baseline at all — relax SLA/M in lean_8gb.yaml"
    for b in baselines:
        assert milp.Q >= b.Q - 1e-9, f"baseline {b.source} beat MILP: {b.Q} > {milp.Q}"
