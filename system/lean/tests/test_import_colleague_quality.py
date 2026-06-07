"""Importer for the colleague's quality_table → lean quality.parquet."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.import_colleague_quality import (
    _agent_to_role_and_domain,
    subset,
    write_with_meta,
)
from src.quality import Quality

_LEAN_ROOT = Path(__file__).resolve().parents[1]
_COLLEAGUE_PARQUET = _LEAN_ROOT.parent / "shared" / "data" / "quality_table.parquet"

_NEEDS_DATA = pytest.mark.skipif(
    not _COLLEAGUE_PARQUET.exists(),
    reason=f"colleague's quality_table missing at {_COLLEAGUE_PARQUET}",
)


def test_agent_to_role_dispatcher():
    assert _agent_to_role_and_domain("A_dispatcher") == ("d", "")


def test_agent_to_role_synth():
    assert _agent_to_role_and_domain("A_synth") == ("y", "")


def test_agent_to_role_specialist_domain_name():
    assert _agent_to_role_and_domain("A_succ_testamentaria") == ("s", "succ_testamentaria")


def test_unrecognised_agent_raises():
    with pytest.raises(ValueError):
        _agent_to_role_and_domain("foo")


@_NEEDS_DATA
def test_subset_filters_to_qwen():
    qt = pd.read_parquet(_COLLEAGUE_PARQUET)
    df, _ = subset(qt, family_prefix="qwen2.5-")
    assert (df["config_id"].str.startswith("qwen2.5-")).all()
    assert df["score"].notna().all()


@_NEEDS_DATA
def test_subset_produces_three_role_codes():
    qt = pd.read_parquet(_COLLEAGUE_PARQUET)
    df, _ = subset(qt, family_prefix="qwen2.5-")
    assert set(df["role"].unique()) == {"d", "s", "y"}


@_NEEDS_DATA
def test_specialist_domains_match_known_legal_set():
    qt = pd.read_parquet(_COLLEAGUE_PARQUET)
    df, _ = subset(qt, family_prefix="qwen2.5-")
    spec = df[df["role"] == "s"]
    domains = set(spec["domain"].unique())
    expected_subset = {
        "succ_testamentaria",
        "succ_legittima",
        "separazione_consensuale",
    }
    assert expected_subset <= domains


@_NEEDS_DATA
def test_scores_in_unit_interval():
    qt = pd.read_parquet(_COLLEAGUE_PARQUET)
    df, _ = subset(qt, family_prefix="qwen2.5-")
    assert df["score"].min() >= 0.0
    assert df["score"].max() <= 1.0 + 1e-9


@_NEEDS_DATA
def test_write_with_meta_round_trips_through_quality(tmp_path):
    qt = pd.read_parquet(_COLLEAGUE_PARQUET)
    df, diagnostics = subset(qt, family_prefix="qwen2.5-")
    out_pq = tmp_path / "quality.parquet"
    out_meta = tmp_path / "quality.meta.json"
    write_with_meta(
        df,
        out_pq,
        out_meta,
        source=_COLLEAGUE_PARQUET,
        diagnostics=diagnostics,
        family_prefix="qwen2.5-",
    )
    q = Quality.from_parquet(out_pq)
    assert len(q.F_d) > 0
    assert len(q.Q_s) > 0
    # Meta sidecar declares provenance.
    import json

    meta = json.loads(out_meta.read_text(encoding="utf-8"))
    assert meta["quality_source"].startswith("colleague:")
    assert meta["source_sha256"]
    assert meta["n_queries_diagnostic"]
