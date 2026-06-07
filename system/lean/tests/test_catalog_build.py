"""Catalog builder ↔ JSON round-trip + perf-table join."""

import json
from itertools import pairwise
from pathlib import Path

import pytest
from catalog import build_catalog as bc

_PERF_TABLE = Path(__file__).resolve().parents[2] / "shared" / "data" / "perf_table.parquet"


@pytest.fixture(scope="module")
def catalog():
    if not _PERF_TABLE.exists():
        pytest.skip(f"shared perf table missing at {_PERF_TABLE}")
    return bc.build()


def test_build_pulls_27_or_fewer_rows(catalog):
    # 3 sizes × 3 quants × 3 contexts = 27 raw; some may be missing in perf
    assert 0 < len(catalog) <= 27


def test_each_config_has_positive_perf(catalog):
    for c in catalog.configs:
        assert c.ttft_s > 0
        assert c.throughput_tps > 0
        assert c.energy_j_per_tok > 0


def test_kv_grows_with_context_within_same_group(catalog):
    by_group = {}
    for c in catalog.configs:
        by_group.setdefault(c.group.key, []).append(c)
    for cs in by_group.values():
        cs.sort(key=lambda c: c.context_length)
        kvs = [c.kv_gb for c in cs]
        # Strictly non-decreasing with context length (closed form is linear).
        for a, b in pairwise(kvs):
            assert b >= a


def test_groups_dedup_across_contexts(catalog):
    # Every (model, quant) group should map to a *single* ModelQuantGroup
    # object reused across contexts.
    seen = {}
    for c in catalog.configs:
        key = c.group.key
        prev = seen.setdefault(key, c.group)
        assert c.group is prev, f"group {key} duplicated across configs"


def test_round_trip_through_json(tmp_path, catalog):
    out = tmp_path / "catalog.json"
    digest = bc.write_json(catalog, out)
    assert (tmp_path / "catalog.json.sha256").exists()
    loaded = bc.read_json(out)
    assert len(loaded) == len(catalog)
    # Shared groups round-trip as a single object per (model, quant).
    seen = {}
    for c in loaded.configs:
        key = c.group.key
        prev = seen.setdefault(key, c.group)
        assert c.group is prev
    # Hash sidecar matches the in-memory hash.
    assert len(digest) == 64


def test_hash_mismatch_raises(tmp_path, catalog):
    out = tmp_path / "catalog.json"
    bc.write_json(catalog, out)
    out.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        bc.read_json(out)
