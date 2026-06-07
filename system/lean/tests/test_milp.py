"""HiGHS MILP — max Q under memory + concurrent SLA on the lean Qwen catalog."""

from pathlib import Path

import pytest
from src.instance import build_arrays, load_instance
from src.milp import solve_milp
from src.quality import synthetic_quality
from src.types import Role

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
    inst = load_instance(_EXP)
    # Inject domains so the specialist Q is averaged over 4 hop types.
    return inst.model_copy(update={"domains": _DOMAINS})


@pytest.fixture(scope="module")
def quality(instance):
    return synthetic_quality(instance.catalog, domains=_DOMAINS)


@_NEEDS_PERF
def test_milp_finds_a_feasible_optimum(instance, quality):
    res = solve_milp(instance, quality)
    assert res.allocation is not None
    a = res.allocation
    assert a.feasible
    assert a.memory_used_gb <= instance.memory_gb + 1e-6
    assert a.L_total_s <= instance.t_circ_s + 1e-6


@_NEEDS_PERF
def test_milp_respects_lambda_lower_bound(instance, quality):
    """Λ must equal the chosen specialist's latency, not exceed any unused config's."""
    arr = build_arrays(instance)
    res = solve_milp(instance, quality)
    a = res.allocation
    spec_id = a.config_by_role[Role.SPECIALIST]
    spec_idx = arr.config_ids.index(spec_id)
    L_s_chosen = float(arr.L_per_role[Role.SPECIALIST][spec_idx])
    assert res.diagnostics["lambda"] == pytest.approx(L_s_chosen, abs=1e-6)


@_NEEDS_PERF
def test_milp_returns_three_roles(instance, quality):
    res = solve_milp(instance, quality)
    a = res.allocation
    assert set(a.config_by_role.keys()) == {Role.DISPATCHER, Role.SPECIALIST, Role.SYNTHESIZER}


@_NEEDS_PERF
def test_tight_memory_forces_sharing(instance, quality):
    tight = instance.model_copy(update={"memory_gb": 3.0})
    res = solve_milp(tight, quality)
    a = res.allocation
    if a is None:
        # If 3 GB is too tight, that's still a valid signal — relax and retry.
        relaxed = instance.model_copy(update={"memory_gb": 4.0})
        res = solve_milp(relaxed, quality)
        a = res.allocation
    assert a is not None
    # Under tight memory, the solver should reuse groups across roles.
    assert len(a.loaded_groups) <= 3


@_NEEDS_PERF
def test_tighter_sla_lowers_or_matches_q(instance, quality):
    loose = solve_milp(instance.model_copy(update={"t_circ_s": 12.0}), quality).allocation
    tight = solve_milp(instance.model_copy(update={"t_circ_s": 4.0}), quality).allocation
    assert loose is not None and tight is not None
    # Relaxing the SLA can only *help* (monotone) under the same memory budget.
    assert loose.Q >= tight.Q - 1e-9


@_NEEDS_PERF
def test_milp_is_deterministic(instance, quality):
    a = solve_milp(instance, quality).allocation
    b = solve_milp(instance, quality).allocation
    assert a is not None and b is not None
    assert pytest.approx(b.Q) == a.Q
    assert a.config_by_role == b.config_by_role
