"""Unit tests for the 'sequential' latency model (paper §6.2 Eq. 7)."""
import math
import pytest
from part1_allocation.optimize.milp import MamapInstance, solve_mamap


def _tiny_instance(M=10.0):
    """One dispatcher, two specialists, one synth; one config each (latencies known)."""
    disp, s1, s2, syn = "A_disp", "A_s1", "A_s2", "A_synth"
    configs = ["cd__c", "cs__c", "cy__c"]
    eligible = {disp: ["cd__c"], s1: ["cs__c"], s2: ["cs__c"], syn: ["cy__c"]}
    # latencies: dispatcher 0.4, specialist (shared) 2.0, synth 3.0
    L = {(disp, "cd__c"): 0.4, (s1, "cs__c"): 2.0, (s2, "cs__c"): 2.0, (syn, "cy__c"): 3.0}
    E = {k: 0.0 for k in L}
    mu = {"cd__c": 1.0, "cs__c": 1.0, "cy__c": 1.0}
    Q = {(disp, "cd__c"): 0.5, (syn, "cy__c"): 0.7}
    Q3 = {(s1, "cs__c", "cd__c"): 0.6, (s2, "cs__c", "cd__c"): 0.6}
    return MamapInstance(specialists=[s1, s2], dispatcher=disp, synthesiser=syn,
                         configs=configs, eligible=eligible, Q=Q, Q3=Q3,
                         L=L, E=E, mu=mu, M=M)


def test_sequential_multiplies_specialist_by_k():
    inst = _tiny_instance()
    # worst_case (concurrent): chain = 0.4 + 2.0 + 3.0 = 5.4  (single specialist)
    sol_wc = solve_mamap(inst, eps=5.4, latency_model="worst_case")
    assert sol_wc.feasible
    assert math.isclose(sol_wc.system_latency, 5.4, abs_tol=1e-6)

    # sequential k=3: chain = 0.4 + 3*2.0 + 3.0 = 9.4
    sol_k3 = solve_mamap(inst, eps=9.4, latency_model="sequential", k_activated=3)
    assert sol_k3.feasible
    assert math.isclose(sol_k3.system_latency, 9.4, abs_tol=1e-6)

    # sequential k=3 must be INFEASIBLE at the concurrent budget 5.4
    sol_inf = solve_mamap(inst, eps=5.4, latency_model="sequential", k_activated=3)
    assert not sol_inf.feasible


def test_sequential_k1_equals_worstcase_here():
    inst = _tiny_instance()
    a = solve_mamap(inst, eps=5.4, latency_model="worst_case")
    b = solve_mamap(inst, eps=5.4, latency_model="sequential", k_activated=1)
    assert a.feasible and b.feasible
    assert math.isclose(a.system_latency, b.system_latency, abs_tol=1e-6)


def test_sequential_rejects_bad_k():
    inst = _tiny_instance()
    with pytest.raises(ValueError):
        solve_mamap(inst, eps=9.4, latency_model="sequential", k_activated=0)


def test_unknown_latency_model_rejected():
    inst = _tiny_instance()
    with pytest.raises(ValueError):
        solve_mamap(inst, eps=9.4, latency_model="parallel_sum")
