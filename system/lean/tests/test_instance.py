"""Instance loader + per-role eligibility + SLA pre-filter."""

from pathlib import Path

import numpy as np
import pytest
from src.instance import build_arrays, load_instance
from src.latency import (
    concurrent_total_latency,
    sequential_total_latency,
    specialist_latency_terms,
)
from src.types import Role

_LEAN_ROOT = Path(__file__).resolve().parents[1]
_EXP = _LEAN_ROOT / "experiments" / "lean_8gb.yaml"
_PERF_TABLE = _LEAN_ROOT.parent / "shared" / "data" / "perf_table.parquet"

_NEEDS_PERF = pytest.mark.skipif(
    not _PERF_TABLE.exists(),
    reason=f"shared perf table missing at {_PERF_TABLE}",
)


@pytest.fixture(scope="module")
def instance():
    # Ensure the catalog is freshly built before loading.
    from catalog import build_catalog as bc

    catalog = bc.build()
    bc.write_json(catalog, _LEAN_ROOT / "catalog" / "catalog.json")
    return load_instance(_EXP)


@_NEEDS_PERF
def test_load_instance_basic_shape(instance):
    assert instance.memory_gb == pytest.approx(6.99)
    assert instance.t_circ_s == pytest.approx(8.0)
    assert len(instance.catalog) > 0


@_NEEDS_PERF
def test_build_arrays_yields_consistent_shapes(instance):
    arr = build_arrays(instance)
    n_k = len(instance.catalog)
    assert arr.weight_gb_of_k.shape == (n_k,)
    assert arr.kv_gb.shape == (n_k,)
    assert arr.group_of_k.shape == (n_k,)
    for role in Role:
        assert arr.L_per_role[role].shape == (n_k,)
        assert arr.E_per_role[role].shape == (n_k,)


@_NEEDS_PERF
def test_eligibility_is_pre_filtered_by_sla(instance):
    arr = build_arrays(instance)
    for role in Role:
        idx = arr.eligibility[role]
        assert idx.size > 0
        # Every eligible config respects the SLA under that role's n_gen.
        assert (arr.L_per_role[role][idx] <= arr.t_circ_s + 1e-9).all()


@_NEEDS_PERF
def test_per_role_n_gen_drives_latency(instance):
    arr = build_arrays(instance)
    # Dispatcher (n_d=15) is faster than synthesizer (n_y=384) for the same config.
    for k in range(len(instance.catalog)):
        assert arr.L_per_role[Role.DISPATCHER][k] < arr.L_per_role[Role.SYNTHESIZER][k]


@_NEEDS_PERF
def test_groups_dedup_by_index(instance):
    arr = build_arrays(instance)
    # group_of_k indexes weights_g — the inverse mapping should be consistent.
    for k, g in enumerate(arr.group_of_k):
        cfg = instance.catalog.configs[k]
        assert arr.group_keys[g] == cfg.group.key


@_NEEDS_PERF
def test_sequential_dominates_concurrent_for_k_ge_2(instance):
    arr = build_arrays(instance)
    k_d, k_s, k_y = 0, 0, 0
    L_conc = concurrent_total_latency(arr, k_d, k_s, k_y)
    L_seq_3 = sequential_total_latency(arr, k_d, k_s, k_y, k_active=3)
    # For k_active=1 they agree.
    L_seq_1 = sequential_total_latency(arr, k_d, k_s, k_y, k_active=1)
    assert L_seq_1 == pytest.approx(L_conc)
    # For k_active=3 the sequential model is strictly more conservative.
    assert L_seq_3 > L_conc


@_NEEDS_PERF
def test_specialist_terms_match_per_role_array(instance):
    arr = build_arrays(instance)
    np.testing.assert_array_equal(
        specialist_latency_terms(arr), arr.L_per_role[Role.SPECIALIST]
    )


@_NEEDS_PERF
def test_too_tight_sla_raises():
    from catalog import build_catalog as bc
    from src.types import Instance

    catalog = bc.build()
    inst = Instance(
        name="tight",
        catalog=catalog,
        memory_gb=6.99,
        t_circ_s=0.01,  # impossible
    )
    with pytest.raises(ValueError, match="empty K_r"):
        build_arrays(inst)
