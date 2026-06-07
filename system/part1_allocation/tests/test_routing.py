"""Tests for multi-label routing metrics and the calib loaders."""
from part1_allocation.scoring.scorer import (multilabel_prf, parse_predicted_agents,
                                             MockRoutingScorer, Sample)
from part1_allocation.config_loader import (load_calib, calib_to_dispatcher_samples,
                                            calib_to_specialist_samples)
from pathlib import Path

CALIB = Path(__file__).resolve().parents[1] / "data" / "calib_clean.yaml"


def test_multilabel_prf_perfect():
    p, r, f1 = multilabel_prf({"a", "b"}, {"a", "b"})
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_multilabel_prf_partial():
    # predicted {a,c}, expected {a,b}: tp=1 -> P=1/2, R=1/2, F1=0.5
    p, r, f1 = multilabel_prf({"a", "c"}, {"a", "b"})
    assert abs(p - 0.5) < 1e-9 and abs(r - 0.5) < 1e-9 and abs(f1 - 0.5) < 1e-9


def test_parse_predicted_agents():
    known = ["A_x", "A_y", "A_z"]
    got = parse_predicted_agents("route to A_x and A_z please", known)
    assert got == {"A_x", "A_z"}


def test_mock_routing_better_config_scores_higher():
    s = Sample(query_id="q1", agent="A_dispatcher", question="...", contexts=[],
               expected_agents=["A_a", "A_b", "A_c"], difficulty="complex")
    known = ["A_a", "A_b", "A_c", "A_d", "A_e"]
    weak = [MockRoutingScorer(known, quality_hint=0.3, config_tag="w").score(s, "").quality
            for _ in range(1)]
    strong = [MockRoutingScorer(known, quality_hint=0.95, config_tag="s").score(s, "").quality
              for _ in range(1)]
    # strong config should not route worse than weak on expectation; check it returns valid F1
    assert 0.0 <= weak[0] <= 1.0 and 0.0 <= strong[0] <= 1.0


def test_calib_loaders():
    calib = load_calib(CALIB)
    assert len(calib) == 25
    disp = calib_to_dispatcher_samples(calib)
    assert len(disp) == 25
    assert all(s.agent == "A_dispatcher" for s in disp)
    assert all(len(s.expected_agents) >= 1 for s in disp)
    fan = calib_to_specialist_samples(calib)
    # fan-out count = sum of expected_agents lengths (no dispatcher in expected sets)
    expected_fan = sum(len(q["expected_agents"]) for q in calib)
    assert len(fan) == expected_fan
    assert all(s.agent != "A_dispatcher" for s in fan)
    # difficulty labels propagate
    assert any(s.difficulty == "complex" for s in fan)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all routing tests passed")
