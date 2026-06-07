"""Tier-2 opt-in modelling options (additive; defaults reproduce v3 exactly).

Covers:
  * Finding 3 -- latency_model="expected_max": the per-query parallel-stage
    latency is the slowest ACTIVE specialist, averaged over queries (the
    architecturally-correct expected latency; NOT the serial Sigma_s pi_s L_s
    that v3 9.1 sketches).
  * Finding 4 -- objective weights and specialist normalisation.

The first test pins the contract that omitting the new kwargs is identical to
the previous behaviour, so these options can never silently change a default run.
"""
import math

import pytest

from part1_allocation.optimize.milp import MamapInstance, solve_mamap


def _inst(per_query_active=None, slow_s3=False):
    roles = ["A_disp", "A_s1", "A_s2", "A_s3", "A_synth"]
    cfg = ["m__q__c4096"]
    L = {(a, "m__q__c4096"): 1.0 for a in roles}
    if slow_s3:
        L[("A_s3", "m__q__c4096")] = 5.0
    Q = {("A_disp", "m__q__c4096"): 0.8, ("A_synth", "m__q__c4096"): 0.7}
    Q3 = {(s, "m__q__c4096", "m__q__c4096"): 0.5 for s in ("A_s1", "A_s2", "A_s3")}
    E = {(a, "m__q__c4096"): 1.0 for a in roles}
    return MamapInstance(
        specialists=["A_s1", "A_s2", "A_s3"], dispatcher="A_disp", synthesiser="A_synth",
        configs=cfg, eligible={a: cfg for a in roles},
        Q=Q, Q3=Q3, L=L, E=E, mu={"m__q__c4096": 1.0}, M=10.0,
        per_query_active=per_query_active,
    )


# --- the no-op contract -----------------------------------------------------
def test_new_kwargs_default_to_v3_behaviour():
    """Omitting the new kwargs must equal the explicit v3 defaults."""
    inst = _inst()
    a = solve_mamap(inst, eps=5.0)
    b = solve_mamap(inst, eps=5.0, latency_model="worst_case",
                    weights=None, normalize_specialists=False)
    assert a.feasible == b.feasible
    assert abs((a.objective or 0.0) - (b.objective or 0.0)) < 1e-12
    assert a.system_latency == b.system_latency
    assert a.allocation == b.allocation


# --- Finding 3: expected_max latency ---------------------------------------
def test_expected_max_amortises_rarely_active_slow_specialist():
    pqa = {"q1": ["A_s1", "A_s2"], "q2": ["A_s1"], "q3": ["A_s2"], "q4": ["A_s3"]}
    inst = _inst(per_query_active=pqa, slow_s3=True)
    wc = solve_mamap(inst, eps=100.0, latency_model="worst_case")
    em = solve_mamap(inst, eps=100.0, latency_model="expected_max")
    # worst_case: disp 1 + max_all_spec 5 + synth 1 = 7
    assert abs(wc.system_latency - 7.0) < 1e-6
    # expected_max: spec stage = mean(max(1,1), 1, 1, 5) = (1+1+1+5)/4 = 2; +1+1 = 4
    assert abs(em.system_latency - 4.0) < 1e-6


def test_expected_max_changes_feasibility():
    pqa = {"q1": ["A_s1", "A_s2"], "q2": ["A_s1"], "q3": ["A_s2"], "q4": ["A_s3"]}
    inst = _inst(per_query_active=pqa, slow_s3=True)
    # eps=5 is infeasible worst-case (needs 7) but feasible expected-max (needs 4)
    assert not solve_mamap(inst, eps=5.0, latency_model="worst_case").feasible
    assert solve_mamap(inst, eps=5.0, latency_model="expected_max").feasible


def test_expected_max_requires_active_sets():
    inst = _inst(per_query_active=None)
    with pytest.raises(ValueError):
        solve_mamap(inst, eps=5.0, latency_model="expected_max")


def test_unknown_latency_model_rejected():
    with pytest.raises(ValueError):
        solve_mamap(_inst(), eps=5.0, latency_model="bogus")


# --- Finding 4: objective weighting / normalisation ------------------------
def _toy_weighting():
    roles = ["A_disp", "A_s1", "A_synth"]
    cfg = ["small", "big"]
    return MamapInstance(
        specialists=["A_s1"], dispatcher="A_disp", synthesiser="A_synth",
        configs=cfg, eligible={a: cfg for a in roles},
        Q={("A_disp", "small"): 0.6, ("A_disp", "big"): 0.9,
           ("A_synth", "small"): 0.5, ("A_synth", "big"): 0.8},
        Q3={("A_s1", "small", "small"): 0.5, ("A_s1", "small", "big"): 0.6,
            ("A_s1", "big", "small"): 0.7, ("A_s1", "big", "big"): 0.9},
        L={(a, c): 1.0 for a in roles for c in cfg},
        E={(a, c): 1.0 for a in roles for c in cfg},
        mu={"small": 1.0, "big": 4.0}, M=20.0,
    )


def test_router_weight_scales_objective_exactly():
    inst = _toy_weighting()
    base = solve_mamap(inst)                       # 0.9 + 0.8 + 0.9 = 2.6
    assert abs(base.objective - 2.6) < 1e-6
    w = solve_mamap(inst, weights=(2.0, 1.0, 1.0))  # 1.8 + 0.8 + 0.9 = 3.5
    assert abs(w.objective - 3.5) < 1e-6


def test_specialist_weight_scales_objective_exactly():
    inst = _toy_weighting()
    w = solve_mamap(inst, weights=(1.0, 1.0, 0.5))  # 0.9 + 0.8 + 0.5*0.9 = 2.15
    assert abs(w.objective - 2.15) < 1e-6


def test_normalize_specialists_divides_block_by_count():
    # Two specialists, each contributing 0.5 -> raw block 1.0, normalised 0.5.
    roles = ["A_disp", "A_s1", "A_s2", "A_synth"]
    cfg = ["m__q__c4096"]
    Q3 = {(s, "m__q__c4096", "m__q__c4096"): 0.5 for s in ("A_s1", "A_s2")}
    inst = MamapInstance(
        specialists=["A_s1", "A_s2"], dispatcher="A_disp", synthesiser="A_synth",
        configs=cfg, eligible={a: cfg for a in roles},
        Q={("A_disp", "m__q__c4096"): 0.8, ("A_synth", "m__q__c4096"): 0.7},
        Q3=Q3, L={(a, "m__q__c4096"): 1.0 for a in roles},
        E={(a, "m__q__c4096"): 1.0 for a in roles},
        mu={"m__q__c4096": 1.0}, M=10.0,
    )
    raw = solve_mamap(inst)                                  # 0.8+0.7 + (0.5+0.5) = 2.5
    norm = solve_mamap(inst, normalize_specialists=True)     # 0.8+0.7 + (1.0/2) = 2.0
    assert abs(raw.objective - 2.5) < 1e-6
    assert abs(norm.objective - 2.0) < 1e-6
