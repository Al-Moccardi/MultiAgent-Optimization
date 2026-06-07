"""rag_bench harness — schema, build_subset stratification, MockGenerator run loop."""


import pytest
from rag_bench.build_subset import from_parquet, load_jsonl, stratify, to_parquet
from rag_bench.eval import (
    MockGenerator,
    _fallback_score,
    average_per_role_config_domain,
    run,
    score_one,
)
from rag_bench.types import Generation, Query, Subset
from src.quality import Quality


def _q(query_id, domain, query="Q?", gold_context="answer A", gold_answer="A"):
    return Query(
        query_id=query_id,
        domain=domain,
        query=query,
        gold_context=gold_context,
        gold_answer=gold_answer,
    )


def test_subset_rejects_empty():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Subset(queries=())


def test_subset_domains_dedup():
    s = Subset(queries=(_q("a", "med"), _q("b", "fin"), _q("c", "med")))
    assert s.domains == ("med", "fin")


def test_stratify_caps_per_domain(tmp_path):
    queries = [_q(f"med_{i}", "medical") for i in range(20)] + [
        _q(f"fin_{i}", "finance") for i in range(5)
    ]
    s = stratify(queries, max_per_domain=3, seed=0)
    by_domain: dict[str, int] = {}
    for q in s.queries:
        by_domain[q.domain] = by_domain.get(q.domain, 0) + 1
    assert by_domain["medical"] == 3
    assert by_domain["finance"] == 3


def test_jsonl_round_trip(tmp_path):
    jsonl = tmp_path / "in.jsonl"
    queries = [_q("a", "med"), _q("b", "fin")]
    with jsonl.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(q.model_dump_json() + "\n")
    loaded = load_jsonl(jsonl)
    assert [q.query_id for q in loaded] == ["a", "b"]


def test_parquet_round_trip(tmp_path):
    out = tmp_path / "subset.parquet"
    subset = Subset(queries=(_q("a", "med"), _q("b", "fin")))
    to_parquet(subset, out)
    loaded = from_parquet(out)
    assert {q.query_id for q in loaded.queries} == {"a", "b"}


def test_mock_generator_produces_generation():
    q = _q("a", "med", gold_context="one two three four five six seven")
    gen = MockGenerator(output_words=4).generate(q, role="s", config_id="x__Q4__c4096")
    assert isinstance(gen, Generation)
    assert gen.answer.split() == ["one", "two", "three", "four"]
    assert gen.role == "s"


def test_fallback_score_in_unit_interval():
    q = _q("a", "med", gold_answer="alpha beta gamma")
    gen = Generation(
        query_id="a",
        domain="med",
        role="s",
        config_id="x__Q4__c4096",
        answer="alpha beta",
        wall_s=0.0,
        n_input_tokens=1,
        n_output_tokens=2,
    )
    s = _fallback_score(gen, q)
    assert 0.0 < s <= 1.0


def test_score_one_handles_missing_shared_scorer(monkeypatch):
    monkeypatch.setattr("rag_bench.eval._shared_scorer", lambda: None)
    q = _q("a", "med", gold_answer="alpha beta gamma")
    gen = MockGenerator(output_words=3).generate(q, role="s", config_id="x__Q4__c4096")
    s = score_one(gen, q)
    assert 0.0 <= s <= 1.0


def test_run_loop_shape():
    subset = Subset(queries=(_q("a", "med"), _q("b", "fin")))
    df = run(
        subset=subset,
        config_ids=["m1__Q4__c4096", "m2__Q5__c4096"],
        generator=MockGenerator(),
    )
    # 3 roles × 2 configs × 2 queries = 12 rows
    assert len(df) == 12
    assert set(df["role"].unique()) == {"d", "s", "y"}


def test_average_collapses_correctly():
    subset = Subset(queries=(_q("a", "med"), _q("b", "med")))
    df = run(
        subset=subset,
        config_ids=["m1__Q4__c4096"],
        generator=MockGenerator(),
    )
    agg = average_per_role_config_domain(df)
    # d + y → one row each; s → one row per (config, domain) = 1 here
    role_counts = agg.groupby("role").size().to_dict()
    assert role_counts == {"d": 1, "s": 1, "y": 1}


def test_aggregated_parquet_loads_into_quality(tmp_path):
    subset = Subset(queries=(_q("a", "med"), _q("b", "fin")))
    df = run(
        subset=subset,
        config_ids=["m1__Q4__c4096"],
        generator=MockGenerator(),
    )
    agg = average_per_role_config_domain(df)
    out = tmp_path / "quality.parquet"
    agg.to_parquet(out)
    q = Quality.from_parquet(out)
    assert "m1__Q4__c4096" in q.F_d
    assert "m1__Q4__c4096" in q.Q_y
    assert ("m1__Q4__c4096", "med") in q.Q_s
    assert ("m1__Q4__c4096", "fin") in q.Q_s
