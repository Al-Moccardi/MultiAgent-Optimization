"""Tier-3 code deliverables (additive; defaults unchanged).

  * Finding 6 -- SpecialistScorer can use a SEPARATE embedder for correctness,
    decoupling it from the retrieval embedder. Default None -> reuse `embedder`,
    so behaviour is unchanged.
  * Finding 8 -- bootstrap_optimum resamples queries and reports allocation
    stability + objective CIs (pure post-processing; no model inference).

Findings 5 (per-specialist ground truth) and 7 (cosine-metric weakness +
human-subset validation) are NOT code changes -- they are dataset/measurement/
paper work and are documented in MAMAP_Tier2_audit.md, not patched here.
"""
import math

import numpy as np
import pandas as pd

from part1_allocation.scoring.scorer import SpecialistScorer, Sample
from shared.schema import AgentSpec, ConfigSpec, DeviceSpec
from part1_allocation.optimize.bootstrap import bootstrap_optimum


class _FakeEmbedder:
    """Deterministic stand-in: maps text -> a fixed unit vector by a tag rule, so
    we can force correctness cosine to differ between two 'embedders'."""
    dim = 4

    def __init__(self, mode):
        self.mode = mode

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(4, dtype="float32")
            # mode 'retriever': everything maps near a single axis (high floor)
            # mode 'distinct': maps by content hash (separates answer vs gt)
            if self.mode == "retriever":
                v[0] = 1.0
            else:
                h = sum(ord(c) for c in t) % 4
                v[h] = 1.0
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.vstack(out)


# --- Finding 6 --------------------------------------------------------------
def test_corr_embedder_defaults_to_main_embedder():
    emb = _FakeEmbedder("retriever")
    sc = SpecialistScorer(embedder=emb)            # no corr_embedder
    assert sc.corr_embedder is emb                 # falls back to the main embedder


def test_separate_corr_embedder_changes_only_correctness():
    sample = Sample(query_id="q::A_s1", agent="A_s1", question="domanda",
                    contexts=["ctx uno"], ground_truth="risposta giusta")
    # With the 'retriever' embedder for everything, correctness is the high-floor 1.0
    same = SpecialistScorer(embedder=_FakeEmbedder("retriever"),
                            include_correctness=True)
    s_same = same.score(sample, answer="una risposta diversa")
    # With a DISTINCT correctness embedder, correctness reflects content difference
    diff = SpecialistScorer(embedder=_FakeEmbedder("retriever"),
                            corr_embedder=_FakeEmbedder("distinct"),
                            include_correctness=True)
    s_diff = diff.score(sample, answer="una risposta diversa")
    # relevancy/faithfulness use the SAME main embedder in both -> unchanged
    assert math.isclose(s_same.answer_relevancy, s_diff.answer_relevancy, rel_tol=1e-6)
    assert math.isclose(s_same.faithfulness, s_diff.faithfulness, rel_tol=1e-6)
    # correctness DOES change because it used the separate embedder
    assert not math.isclose(s_same.correctness, s_diff.correctness, rel_tol=1e-6)


# --- Finding 8 (bootstrap) --------------------------------------------------
def _toy_tables(flip=True):
    """1 disp, 1 specialist (A_s1), 1 synth, 2 configs over 5 queries.
    If flip=True, A_s1's better config alternates by query -> unstable optimum."""
    agents = [AgentSpec("A_dispatcher", "d", "", 2048, True),
              AgentSpec("A_s1", "s", "", 2048),
              AgentSpec("A_synth", "y", "", 2048)]
    configs = {
        "a__Q4_K_M__c4096": ConfigSpec("a__Q4_K_M__c4096", "a", "h", "Q4_K_M", 4096, "", 1.0),
        "b__Q4_K_M__c4096": ConfigSpec("b__Q4_K_M__c4096", "b", "h", "Q4_K_M", 4096, "", 1.0),
    }
    dev = DeviceSpec("t", "DISCRETE_GPU", 8.0, 10.0, "mean", "none")
    rows = []
    for i in range(5):
        qid = f"Q{i}"
        for cid in configs:
            rows.append(dict(agent="A_dispatcher", query_id=qid, config_id=cid,
                             output="A_s1", n_out_tokens=10, quality=0.8, confidence=0.8,
                             difficulty="", risk_level="", category="",
                             expected_agents="A_s1", q_faithfulness=np.nan,
                             q_answer_relevancy=np.nan, q_context_precision=np.nan,
                             q_context_recall=np.nan, q_correctness=np.nan))
        qa, qb = (0.62, 0.58) if (i % 2 == 0 or not flip) else (0.58, 0.62)
        for cid, val in [("a__Q4_K_M__c4096", qa), ("b__Q4_K_M__c4096", qb)]:
            rows.append(dict(agent="A_s1", query_id=f"{qid}::A_s1", config_id=cid,
                             output="x", n_out_tokens=40, quality=val, confidence=0.7,
                             difficulty="", risk_level="", category="",
                             expected_agents="A_s1", q_faithfulness=0.6,
                             q_answer_relevancy=0.6, q_context_precision=1.0,
                             q_context_recall=1.0, q_correctness=np.nan))
        for cid in configs:
            rows.append(dict(agent="A_synth", query_id=f"{qid}::A_synth", config_id=cid,
                             output="f", n_out_tokens=50, quality=0.5, confidence=0.7,
                             difficulty="", risk_level="", category="",
                             expected_agents="A_s1", q_faithfulness=0.6,
                             q_answer_relevancy=0.6, q_context_precision=1.0,
                             q_context_recall=1.0, q_correctness=0.5))
    qdf = pd.DataFrame(rows)
    perf = pd.DataFrame([dict(hardware="t", config_id=c, peak_mem_gb=1.0, ttft_s=0.2,
                              throughput_tok_s=40, energy_j_per_tok=0.5) for c in configs])
    return agents, configs, dev, qdf, perf


def test_bootstrap_runs_and_reports_structure():
    agents, configs, dev, qdf, perf = _toy_tables(flip=False)
    s = bootstrap_optimum(qdf, perf, agents, configs, dev, "t", n_boot=50, eps=None, seed=1)
    assert s["n_feasible"] > 0
    assert "objective_ci" in s and len(s["objective_ci"]) == 2
    assert "per_agent_stability" in s
    assert "A_s1" in s["per_agent_stability"]


def test_bootstrap_detects_unstable_optimum():
    # flip=True makes A_s1's argmax config genuinely depend on the query sample
    agents, configs, dev, qdf, perf = _toy_tables(flip=True)
    s = bootstrap_optimum(qdf, perf, agents, configs, dev, "t", n_boot=300, eps=None, seed=7)
    st = s["per_agent_stability"]["A_s1"]
    # both configs should appear as winners across resamples, and the objective
    # should vary -- the whole point of the bootstrap.
    assert st["distinct_configs_seen"] == 2
    assert st["selection_freq"] < 1.0
    assert s["objective_max"] > s["objective_min"]


def test_bootstrap_resampling_is_not_a_noop():
    """Regression: an earlier version suffixed duplicate draws on the wrong side
    of '::', so derive's per-query mean collapsed them and resampling did nothing
    (zero objective variance). A stable instance gives a near-degenerate CI; an
    unstable one must give a NON-degenerate spread."""
    agents, configs, dev, qdf, perf = _toy_tables(flip=True)
    s = bootstrap_optimum(qdf, perf, agents, configs, dev, "t", n_boot=300, eps=None, seed=11)
    assert s["objective_max"] - s["objective_min"] > 1e-6
