"""Frozen models — schema + KV closed form + group dedup."""

import pytest
from pydantic import ValidationError
from src.types import (
    Allocation,
    Catalog,
    Config,
    Instance,
    ModelQuantGroup,
    Role,
)


def _group(**overrides):
    base = dict(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        quant="Q5_K_M",
        params=3_085_938_688,
        weight_gb=2.1,
        c_max=32768,
        n_layers=36,
        n_kv_heads=2,
        head_dim=128,
    )
    base.update(overrides)
    return ModelQuantGroup(**base)


def _cfg(group, ctx=4096, **overrides):
    base = dict(
        group=group,
        context_length=ctx,
        ttft_s=0.03,
        throughput_tps=70.0,
        energy_j_per_tok=0.42,
    )
    base.update(overrides)
    return Config(**base)


# --- ModelQuantGroup ---------------------------------------------------------


def test_kv_closed_form_matches_formula():
    g = _group(n_layers=4, n_kv_heads=2, head_dim=64, kv_dtype_bytes=2.0)
    # κ = 2·L·n_kv·d·c·b_kv bytes
    assert g.kv_bytes(4096) == 2 * 4 * 2 * 64 * 4096 * 2


def test_kv_scales_linearly_with_context():
    g = _group()
    assert g.kv_bytes(8192) == 2 * g.kv_bytes(4096)


def test_group_rejects_nonpositive_fields():
    with pytest.raises(ValidationError):
        _group(weight_gb=0)
    with pytest.raises(ValidationError):
        _group(n_layers=-1)


# --- Config -----------------------------------------------------------------


def test_config_latency_and_energy_formulas():
    g = _group()
    c = _cfg(g, ttft_s=0.05, throughput_tps=100.0, energy_j_per_tok=0.3)
    assert c.latency(200) == pytest.approx(0.05 + 200 / 100)
    assert c.energy(200) == pytest.approx(0.3 * 200)


def test_config_rejects_ctx_above_cmax():
    g = _group(c_max=4096)
    with pytest.raises(ValidationError):
        _cfg(g, ctx=8192)


def test_config_id_uses_explicit_perf_prefix_when_set():
    g = _group(
        model_id="Qwen/Qwen2.5-3B-Instruct", quant="Q5_K_M", perf_prefix="qwen2.5-3b"
    )
    c = _cfg(g, ctx=4096)
    # When perf_prefix is set, config_id matches the colleague's table verbatim.
    assert c.config_id == "qwen2.5-3b__Q5_K_M__c4096"


def test_config_id_falls_back_to_derivation_without_perf_prefix():
    g = _group(model_id="Qwen/Qwen2.5-3B-Instruct", quant="Q5_K_M")  # no perf_prefix
    c = _cfg(g, ctx=4096)
    # Derived: split('/')[-1].lower().replace('-instruct', '')
    assert c.config_id == "qwen2.5-3b__Q5_K_M__c4096"


# --- Catalog -----------------------------------------------------------------


def test_catalog_dedups_groups_by_key():
    g1 = _group(model_id="Qwen/Qwen2.5-3B-Instruct", quant="Q5_K_M")
    g2 = _group(model_id="Qwen/Qwen2.5-1.5B-Instruct", quant="Q5_K_M", params=1_543_714_304)
    cat = Catalog(
        configs=(_cfg(g1, ctx=4096), _cfg(g1, ctx=8192), _cfg(g2, ctx=4096)),
    )
    assert len(cat) == 3
    keys = {g.key for g in cat.groups}
    assert keys == {g1.key, g2.key}


def test_catalog_rejects_duplicate_keys():
    g = _group()
    with pytest.raises(ValidationError):
        Catalog(configs=(_cfg(g, ctx=4096), _cfg(g, ctx=4096)))


# --- Role + Instance + Allocation -------------------------------------------


def test_role_enum_values():
    assert Role.DISPATCHER.value == "d"
    assert Role.SPECIALIST.value == "s"
    assert Role.SYNTHESIZER.value == "y"


def test_instance_requires_non_empty_catalog():
    with pytest.raises(ValidationError):
        Instance(name="x", catalog=Catalog(configs=()), memory_gb=8.0, t_circ_s=4.0)


def test_allocation_round_trip():
    a = Allocation(
        instance_name="t",
        config_by_role={
            Role.DISPATCHER: "qwen2_5-0_5b__Q3_K_M__c2048",
            Role.SPECIALIST: "qwen2_5-3b__Q5_K_M__c4096",
            Role.SYNTHESIZER: "qwen2_5-1_5b__Q5_K_M__c4096",
        },
        loaded_groups=(
            ("Qwen/Qwen2.5-0.5B-Instruct", "Q3_K_M"),
            ("Qwen/Qwen2.5-3B-Instruct", "Q5_K_M"),
            ("Qwen/Qwen2.5-1.5B-Instruct", "Q5_K_M"),
        ),
        Q=0.72,
        L_total_s=5.2,
        memory_used_gb=4.8,
        feasible=True,
        source="milp",
    )
    assert pytest.approx(0.72) == a.Q
    assert a.feasible
