"""Tests for the specialist (RAGAS-style) evaluation path: corpus store, gold
loaders, the offline JudgeScorer, and the derive NaN guard."""
import math

from part1_allocation.scoring.corpus import CorpusStore
from part1_allocation.scoring.scorer import (Sample, JudgeScorer, MockJudgeBackend,
                                             aggregate)


def test_corpus_resolves_gold_ids():
    store = CorpusStore({"IT_CC_art_602": "Art. 602 ...", "ECLI:IT:X:1": "case text"})
    ctx = store.contexts_for(["IT_CC_art_602", "MISSING"], ["ECLI:IT:X:1"])
    assert ctx == ["Art. 602 ...", "case text"]      # laws first, missing skipped
    assert abs(store.coverage(["IT_CC_art_602", "MISSING"]) - 0.5) < 1e-9


def test_judge_correctness_only_without_contexts():
    # No contexts -> grounding metrics NaN, but correctness + relevancy present.
    judge = MockJudgeBackend({"correctness": 0.8, "answer_relevancy": 0.9,
                              "faithfulness": None, "context_precision": None,
                              "context_recall": None})
    s = Sample(query_id="q1", agent="A_x", question="?", contexts=[],
               ground_truth="gold")
    sc = JudgeScorer(judge).score(s, "an answer")
    assert math.isnan(sc.faithfulness) and math.isnan(sc.context_precision)
    assert abs(sc.correctness - 0.8) < 1e-9
    # quality blends correctness (0.5) + relevancy-only grounding (0.9) = 0.85
    assert abs(sc.quality - aggregate(answer_relevancy=0.9, correctness=0.8)) < 1e-9


def test_judge_full_metrics_with_contexts():
    judge = MockJudgeBackend({"correctness": 0.7, "faithfulness": 0.8,
                              "context_precision": 0.75, "context_recall": 0.6,
                              "answer_relevancy": 0.9})
    s = Sample(query_id="q2", agent="A_x", question="?",
               contexts=["ctx a", "ctx b"], ground_truth="gold")
    sc = JudgeScorer(judge).score(s, "an answer")
    for v in (sc.correctness, sc.faithfulness, sc.context_precision,
              sc.context_recall, sc.answer_relevancy):
        assert not math.isnan(v)
    assert 0.0 <= sc.quality <= 1.0


def test_judge_unparseable_output_is_nan_not_crash():
    class BadJudge:
        def generate(self, *a, **k):
            from part1_allocation.inference.backend import GenResult
            return GenResult(text="sorry, I cannot comply", n_out_tokens=5,
                             ttft_s=0.0, total_s=0.0)
        def close(self): pass
    s = Sample(query_id="q3", agent="A_x", question="?", contexts=[], ground_truth="g")
    sc = JudgeScorer(BadJudge()).score(s, "ans")
    assert math.isnan(sc.quality)        # unparseable -> NaN, no exception


def test_specialist_scorer_embedder_only():
    # No judge, only an embedder: correctness (vs ground_truth) + answer_relevancy
    # are computed from embeddings; grounding metrics stay NaN.
    from part1_allocation.scoring.scorer import SpecialistScorer
    from part1_allocation.scoring.embeddings import HashingEmbedder
    emb = HashingEmbedder(dim=256)
    s = Sample(query_id="q", agent="A_x",
               question="conditions for a valid holographic will",
               contexts=[], ground_truth="a holographic will needs handwriting, date, signature")
    sc = SpecialistScorer(judge_backend=None, embedder=emb).score(
        s, "a holographic will must be handwritten, dated and signed")
    assert not math.isnan(sc.correctness)        # embedding similarity to ground truth
    assert not math.isnan(sc.answer_relevancy)   # embedding similarity to question
    assert math.isnan(sc.faithfulness)           # no judge/contexts -> NaN
    assert 0.0 <= sc.quality <= 1.0


def test_specialist_scorer_judge_plus_embedder():
    from part1_allocation.scoring.scorer import SpecialistScorer
    from part1_allocation.scoring.embeddings import HashingEmbedder
    judge = MockJudgeBackend({"correctness": 0.7, "faithfulness": 0.8,
                              "context_precision": 0.75, "context_recall": 0.6,
                              "answer_relevancy": 0.5})
    s = Sample(query_id="q", agent="A_x", question="?",
               contexts=["ctx"], ground_truth="g")
    sc = SpecialistScorer(judge_backend=judge, embedder=HashingEmbedder(128)).score(s, "ans")
    # grounding from judge, relevancy from embedder (overrides judge's 0.5)
    assert abs(sc.faithfulness - 0.8) < 1e-9
    assert not math.isnan(sc.answer_relevancy)
    assert 0.0 <= sc.quality <= 1.0


def test_faiss_build_and_retrieve_roundtrip(tmp_path=None):
    import tempfile, os
    from part1_allocation.scoring.embeddings import HashingEmbedder
    from part1_allocation.scoring.retrieval import build_index, FaissRetriever
    corpus = {"d1": "holographic will handwriting date signature",
              "d2": "community property regime joint assets",
              "d3": "consensual separation agreement court ratification"}
    emb = HashingEmbedder(dim=256)
    d = tempfile.mkdtemp()
    build_index(corpus, emb, os.path.join(d, "x.faiss"), os.path.join(d, "x.manifest.json"))
    r = FaissRetriever(emb, corpus, k=2).add(os.path.join(d, "x.faiss"),
                                             os.path.join(d, "x.manifest.json"))
    hits = r.retrieve("requirements for a valid holographic will", k=2)
    assert len(hits) == 2 and corpus["d1"] in hits   # most relevant retrieved


def test_specialist_scorer_judge_free_full_metrics():
    # Judge-free: all 5 metrics computed without any LLM, just from gold + retrieval.
    from part1_allocation.scoring.scorer import SpecialistScorer
    from part1_allocation.scoring.embeddings import HashingEmbedder
    emb = HashingEmbedder(dim=512)
    s = Sample(query_id="q1", agent="A_x", question="holographic will validity",
               contexts=["a holographic will needs handwriting date signature",
                         "intestate succession order"],
               ground_truth="autografia date signature are required",
               gold_law_ids=["L1", "L2"], gold_case_ids=["C1"],
               retrieved_ids=["L1", "X9", "C1"])     # 2 of 3 hits = gold
    sc = SpecialistScorer(embedder=emb).score(
        s, "an olographic will requires handwriting, a date and a signature")
    assert not math.isnan(sc.correctness)
    assert not math.isnan(sc.answer_relevancy)
    assert not math.isnan(sc.faithfulness)
    # id-based: |retr ∩ gold| / |retr| = 2/3 ; / |gold| = 2/3
    assert abs(sc.context_precision - 2/3) < 1e-9
    assert abs(sc.context_recall - 2/3) < 1e-9
    assert 0.0 <= sc.quality <= 1.0


def test_context_metrics_perfect_when_oracle_contexts():
    # No retriever -> retrieved_ids=[] -> trivially perfect P/R (oracle convention).
    from part1_allocation.scoring.scorer import SpecialistScorer
    from part1_allocation.scoring.embeddings import HashingEmbedder
    s = Sample(query_id="q2", agent="A_x", question="q",
               contexts=["gold ctx"], ground_truth="gt",
               gold_law_ids=["L1"], retrieved_ids=[])
    sc = SpecialistScorer(embedder=HashingEmbedder(256)).score(s, "an answer")
    assert sc.context_precision == 1.0 and sc.context_recall == 1.0


def test_quality_csv_export():
    import os, tempfile
    import pandas as pd
    from part1_allocation.measure.quality_csv import export_quality_csv
    df = pd.DataFrame([
        {"agent":"A_x","config_id":"m1","query_id":"q1","quality":0.8,
         "q_faithfulness":0.9,"q_answer_relevancy":0.7,"q_context_precision":1.0,
         "q_context_recall":0.5,"q_correctness":0.8,"n_out_tokens":50,"output":""},
        {"agent":"A_x","config_id":"m1","query_id":"q2","quality":0.6,
         "q_faithfulness":0.7,"q_answer_relevancy":0.5,"q_context_precision":0.5,
         "q_context_recall":0.5,"q_correctness":0.6,"n_out_tokens":40,"output":""},
        {"agent":"A_y","config_id":"m1","query_id":"q3","quality":0.4,
         "q_faithfulness":0.5,"q_answer_relevancy":0.4,"q_context_precision":0.3,
         "q_context_recall":0.4,"q_correctness":0.4,"n_out_tokens":40,"output":""},
    ])
    d = tempfile.mkdtemp(); p = os.path.join(d, "scorecard.csv")
    agg = export_quality_csv(df, p)
    assert os.path.exists(p)
    row = agg[(agg.agent=="A_x") & (agg.config_id=="m1")].iloc[0]
    assert abs(row["quality"] - 0.7) < 1e-9
    assert int(row["n_queries"]) == 2
    assert list(agg.columns)[:3] == ["agent","config_id","n_queries"]


def test_derive_drops_nan_quality_instead_of_crashing():
    # NaN quality on (agent, config) must be dropped, not crash the MILP.
    import pandas as pd
    from part1_allocation.optimize.derive import build_instance
    from shared.schema import AgentSpec, ConfigSpec, DeviceSpec

    # need: dispatcher, at least one specialist, synth
    agents = [
        AgentSpec(agent_id="A_dispatcher", name="d", description="", c_min=2048,
                  is_dispatcher=True),
        AgentSpec(agent_id="A_x", name="x", description="", c_min=4096),
        AgentSpec(agent_id="A_synth", name="s", description="", c_min=4096),
    ]
    configs = {
        "m__Q4__c4096": ConfigSpec("m__Q4__c4096", "m", "", "Q4", 4096, "p", 1.0),
        "n__Q4__c4096": ConfigSpec("n__Q4__c4096", "n", "", "Q4", 4096, "p", 1.0),
    }
    perf = pd.DataFrame([
        {"hardware": "h", "config_id": cid, "peak_mem_gb": 1.0,
         "ttft_s": 0.1, "throughput_tok_s": 50.0, "energy_j_per_tok": float("nan")}
        for cid in configs])
    # Each agent must have at least one finite-quality row to be usable.
    rows = []
    for a in ("A_dispatcher", "A_x", "A_synth"):
        for cid, q in (("m__Q4__c4096", float("nan")),  # NaN must be dropped
                       ("n__Q4__c4096", 0.6)):
            rows.append({"agent": a, "config_id": cid, "query_id": f"q::{a}",
                         "quality": q, "n_out_tokens": 100, "output": ""})
    quality = pd.DataFrame(rows)
    dev = DeviceSpec(name="h", hw_class="CPU_ONLY", memory_budget_gb=10.0,
                     latency_sla_s=8.0)
    inst = build_instance(quality, perf, agents, configs, dev, hardware="h")
    assert "m__Q4__c4096" not in inst.eligible["A_x"]
    assert inst.eligible["A_x"] == ["n__Q4__c4096"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all specialist tests passed")


def test_include_correctness_flag_excludes_from_Q_but_keeps_in_record():
    """v2.1 policy: specialist's Q should exclude correctness (because gold GT
    covers the full composed answer across multiple agents), but q_correctness
    must still be measured and saved to the parquet for post-hoc ablations."""
    from part1_allocation.scoring.scorer import Sample, SpecialistScorer
    import numpy as np

    class FakeEmbedder:
        def __init__(self): self.dim = 4
        def encode(self, texts):
            out = []
            for t in texts:
                v = np.array([0.5 + 0.1*len(t), 0.6, 0.7, 0.8])
                out.append(v / np.linalg.norm(v))
            return out

    emb = FakeEmbedder()
    s = Sample(query_id='q1', agent='A_x', question='Q',
               contexts=['ctx text'],
               ground_truth='long ground truth that differs',
               retrieved_ids=['L1'], gold_law_ids=['L1'], gold_case_ids=[])

    sc_spec = SpecialistScorer(judge_backend=None, embedder=emb,
                               include_correctness=False)
    sc_syn = SpecialistScorer(judge_backend=None, embedder=emb,
                              include_correctness=True)
    qs_spec = sc_spec.score(s, 'a useful response')
    qs_syn = sc_syn.score(s, 'a useful response')

    # MEASUREMENT IDENTITY: both scorers compute the same per-metric values
    assert qs_spec.correctness == qs_syn.correctness, "correctness measurement must be identical"
    assert qs_spec.faithfulness == qs_syn.faithfulness
    assert qs_spec.answer_relevancy == qs_syn.answer_relevancy

    # AGGREGATE DIFFERS: specialist excludes correctness from Q
    assert qs_spec.quality != qs_syn.quality, "aggregates must differ"
    # specialist Q = mean(faithfulness, relevancy), no correctness
    expected_spec = (qs_spec.faithfulness + qs_spec.answer_relevancy) / 2
    assert abs(qs_spec.quality - expected_spec) < 1e-6
