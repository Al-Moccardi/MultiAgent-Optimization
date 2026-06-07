"""Regression tests for the Pareto epsilon-grid scale (Finding #2).

The latency constraint (milp.py, eq. 8) bounds the per-query inference CHAIN

    L_router + L_max_spec + L_synth  <=  eps

i.e. the SUM of three roles. The epsilon grid must live on that same system
scale. A grid built on a single agent's latency range (the historical bug)
lies BELOW the feasible system latency whenever specialists + synth are
present, so the entire sweep collapses onto the unconstrained point and the
quality-vs-latency frontier is never traced.

These tests pin the corrected behaviour and would FAIL against the old
per-agent-scale grid.
"""
import math

from part1_allocation.optimize.milp import MamapInstance, solve_mamap
from part1_allocation.optimize.pareto import latency_grid, build_frontier


def _toy_inst():
    """1 dispatcher, 2 specialists, 1 synth, 2 configs (2.0s vs 3.0s each).

    Single-agent latency range is [2.0, 3.0]; but the binding system chain is
    router + slowest-specialist + synth, whose range is [6.0, 9.0]. This gap is
    exactly what the old grid got wrong.
    """
    roles = ["A_disp", "A_s1", "A_s2", "A_synth"]
    configs = ["m__q__c4096", "m2__q__c4096"]
    L = {(a, "m__q__c4096"): 2.0 for a in roles}
    L.update({(a, "m2__q__c4096"): 3.0 for a in roles})
    Q = {("A_disp", "m__q__c4096"): 0.8, ("A_disp", "m2__q__c4096"): 0.9,
         ("A_synth", "m__q__c4096"): 0.7, ("A_synth", "m2__q__c4096"): 0.85}
    Q3 = {(s, ka, kd): (0.5 if "m2" in ka else 0.4)
          for s in ("A_s1", "A_s2") for ka in configs for kd in configs}
    E = {(a, c): 1.0 for a in roles for c in configs}
    mu = {c: 1.0 for c in configs}
    return MamapInstance(
        specialists=["A_s1", "A_s2"], dispatcher="A_disp", synthesiser="A_synth",
        configs=configs, eligible={a: configs for a in roles},
        Q=Q, Q3=Q3, L=L, E=E, mu=mu, M=100.0,
    )


def test_grid_is_on_system_latency_scale_not_per_agent():
    """The grid must span the sum-of-three-roles range, not a single agent's."""
    inst = _toy_inst()
    grid = latency_grid(inst, n=5)
    # Per-agent max is 3.0; the system chain min is 6.0. A correct grid starts
    # at/above the system minimum, well beyond any single-agent latency.
    assert min(grid) >= 3.0 + 1e-9, (
        f"grid lower bound {min(grid)} is on the per-agent scale; "
        f"expected the system-chain scale (>= 6.0)")
    # Analytic system bounds for this instance: lo=2+2+2=6, hi=3+3+3=9.
    assert abs(min(grid) - 6.0) < 1e-9
    assert abs(max(grid) - 9.0) < 1e-9


def test_grid_brackets_the_unconstrained_system_latency():
    """The unconstrained optimum's system latency must lie within the grid."""
    inst = _toy_inst()
    grid = latency_grid(inst, n=5)
    unc = solve_mamap(inst, eps=None)
    assert unc.feasible
    assert min(grid) - 1e-9 <= unc.system_latency <= max(grid) + 1e-9


def test_frontier_is_traced_not_collapsed():
    """With a correct grid the sweep yields multiple non-dominated points that
    trade quality against system latency. The old per-agent grid produced only
    the unconstrained point (all capped solves were infeasible)."""
    inst = _toy_inst()
    front = build_frontier(inst, n=5)
    assert len(front) >= 2, (
        "frontier collapsed to a single point -- the eps grid is not biting the "
        "system-latency constraint")
    # latency strictly increases and quality is non-decreasing along the front
    lats = [s.system_latency for _, s in front]
    quals = [s.objective for _, s in front]
    assert lats == sorted(lats)
    assert quals == sorted(quals)
    # the cheapest point must respect a tighter latency than the richest
    assert lats[0] < lats[-1]
    assert quals[0] < quals[-1] + 1e-9
