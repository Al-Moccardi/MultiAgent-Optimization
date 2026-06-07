"""Tests for the end-to-end MAMAP MILP (v2.1)."""
import math

import pulp

from part1_allocation.optimize.milp import MamapInstance, solve_mamap


def _toy_inst(eps_loadable=True):
    """Tiny instance: 1 specialist, 1 dispatcher, 1 synth, 2 configs."""
    return MamapInstance(
        specialists=["S1"],
        dispatcher="DISP",
        synthesiser="SYN",
        configs=["small", "big"],
        eligible={"S1": ["small", "big"], "DISP": ["small", "big"], "SYN": ["small", "big"]},
        Q={("DISP", "small"): 0.6, ("DISP", "big"): 0.9,
           ("SYN", "small"): 0.5, ("SYN", "big"): 0.8},
        Q3={("S1", "small", "small"): 0.5, ("S1", "small", "big"): 0.6,
            ("S1", "big",   "small"): 0.7, ("S1", "big",   "big"): 0.9},
        L={("S1", "small"): 1.0, ("S1", "big"): 3.0,
           ("DISP", "small"): 0.2, ("DISP", "big"): 0.5,
           ("SYN", "small"): 0.8, ("SYN", "big"): 2.0},
        E={("S1", "small"): 1.0, ("S1", "big"): 3.0,
           ("DISP", "small"): 0.2, ("DISP", "big"): 0.5,
           ("SYN", "small"): 0.8, ("SYN", "big"): 2.0},
        mu={"small": 1.0, "big": 4.0}, M=20.0 if eps_loadable else 5.5,
    )


def test_end_to_end_picks_best_when_unconstrained():
    inst = _toy_inst()
    sol = solve_mamap(inst)
    assert sol.feasible
    assert sol.allocation == {"DISP": "big", "S1": "big", "SYN": "big"}
    # objective = Q[DISP,big] + Q[SYN,big] + Q3[S1,big,big] = 0.9 + 0.8 + 0.9 = 2.6
    assert abs(sol.objective - 2.6) < 1e-6


def test_bilinear_gating_depends_on_dispatcher_choice():
    """If we force DISP=small, the specialist's gated Q drops accordingly."""
    inst = _toy_inst()
    # add a constraint via tightening: only "small" eligible for DISP
    inst.eligible["DISP"] = ["small"]
    inst.Q = {(a, k): v for (a, k), v in inst.Q.items()
              if not (a == "DISP" and k not in inst.eligible["DISP"])}
    sol = solve_mamap(inst)
    assert sol.feasible
    assert sol.allocation["DISP"] == "small"
    # Q3[S1, big, small] = 0.7, Q[DISP, small] = 0.6, Q[SYN, big] = 0.8
    assert abs(sol.objective - (0.6 + 0.8 + 0.7)) < 1e-6


def test_system_latency_constraint_chains_router_max_synth():
    inst = _toy_inst()
    # eps = 0.2 (DISP small) + 1.0 (S1 small) + 0.8 (SYN small) = 2.0 -> all small allowed
    sol = solve_mamap(inst, eps=2.0)
    assert sol.feasible
    assert sol.allocation == {"DISP": "small", "S1": "small", "SYN": "small"}
    # eps = 1.9 -> infeasible
    sol2 = solve_mamap(inst, eps=1.9)
    assert not sol2.feasible


def test_memory_budget_forces_small():
    inst = _toy_inst()
    inst.M = 3.0    # cannot load any "big"
    sol = solve_mamap(inst)
    assert sol.feasible
    assert all(k == "small" for k in sol.allocation.values())


def test_one_context_per_group_via_naming():
    """Two configs sharing the (model,quant) prefix collapse to one loaded."""
    inst = MamapInstance(
        specialists=["S1"], dispatcher="DISP", synthesiser="SYN",
        configs=["m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096"],
        eligible={"S1": ["m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096"],
                  "DISP": ["m__Q4__c4096", "n__Q4__c4096"],
                  "SYN":  ["m__Q4__c4096", "n__Q4__c4096"]},
        Q={("DISP", "m__Q4__c4096"): 0.5, ("DISP", "n__Q4__c4096"): 0.5,
           ("SYN", "m__Q4__c4096"): 0.5, ("SYN", "n__Q4__c4096"): 0.5},
        Q3={("S1", k_a, k_d): 0.5 for k_a in ["m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096"]
            for k_d in ["m__Q4__c4096", "n__Q4__c4096"]},
        L={(a, k): 1.0 for a in ("DISP", "S1", "SYN")
           for k in ("m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096")
           if k in ["m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096"]},
        E={(a, k): 1.0 for a in ("DISP", "S1", "SYN")
           for k in ("m__Q4__c4096", "m__Q4__c8192", "n__Q4__c4096")},
        mu={"m__Q4__c4096": 5.0, "m__Q4__c8192": 5.0, "n__Q4__c4096": 5.0},
        M=11.0,  # would fit 2 of 3 if independent; (6b) caps m__ to 1 instance
    )
    sol = solve_mamap(inst)
    assert sol.feasible
    # m's two contexts must NOT both be loaded
    loaded_m = [k for k in sol.loaded if k.startswith("m__Q4__")]
    assert len(loaded_m) <= 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("all MILP v2.1 tests passed")
