"""
part1_allocation/optimize/milp.py
=================================
The MAMAP MILP, end-to-end formulation (v2.1, judge-free + router-gated).

Roles:
  A_disp = single dispatcher agent (provides routing decisions)
  A_spec = specialist agents (gated by router)
  A_synth = single synthesiser agent (linear quality, after specialists)

Objective:
  max  Q[A_disp, k_d]*x[A_disp,k_d]              # router F1
     + Q[A_synth, k_s]*x[A_synth,k_s]            # synth final-answer quality
     + sum_{a in A_spec, k_a, k_d} Q3[a,k_a,k_d] * z[a,k_a,k_d]
       with z[a,k_a,k_d] = x[a,k_a] * x[A_disp,k_d]  (linearised below)

The bilinear Q3[a,k_a,k_d] = E_q[1{a in router_pred(k_d, q)} * Q_gen(a,k_a,q)]
is computed offline from the parquet (see optimize/derive.py); the MILP sees it
as a 3-index numeric table.

Constraints:
  (4)  sum_k x[a,k] = 1                              for each agent
  (5)  x[a,k] <= y[k]
  (6)  sum_k mu[k]*y[k] <= M                                                    (memory)
  (6b) sum_{k in group g} y[k] <= 1                  for groups g=(m,q)
  (7)  L[A_disp,k_d]*x_d + L_max_spec + L[A_synth,k_s]*x_s <= eps               (sys latency)
       L_max_spec >= sum_k L[s,k]*x[s,k]             for each specialist s
  z:   z <= x[a,k_a], z <= x[A_disp,k_d], z >= x[a,k_a]+x[A_disp,k_d]-1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import pulp


def _auto_groups(configs: list[str]) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for k in configs:
        g.setdefault(k.rsplit("__", 1)[0], []).append(k)
    return g


@dataclass
class MamapInstance:
    specialists: list[str]
    dispatcher: str
    synthesiser: str

    configs: list[str]
    eligible: dict[str, list[str]]

    # quality: linear (dispatcher, synth) and gated (specialists)
    Q: dict[tuple[str, str], float]              # (a,k) for dispatcher & synth
    Q3: dict[tuple[str, str, str], float]        # (specialist, k_a, k_d)

    # performance (per agent x k)
    L: dict[tuple[str, str], float]
    E: dict[tuple[str, str], float]
    mu: dict[str, float]
    M: float

    groups: Optional[dict[str, list[str]]] = None

    # Per-query active specialist sets, qid -> [specialist_id, ...].
    # OPTIONAL: only consumed by the `expected_max` latency model (Finding 3).
    # When None (default), the latency constraint is the worst-case max over ALL
    # specialists (v3 §9.1 default) and nothing here changes.
    per_query_active: Optional[dict[str, list[str]]] = None

    @property
    def agents(self) -> list[str]:
        return [self.dispatcher] + list(self.specialists) + [self.synthesiser]

    def validate(self) -> None:
        for a in self.agents:
            if not self.eligible.get(a):
                raise ValueError(f"Agent '{a}' has no eligible config.")
        for a in [self.dispatcher, self.synthesiser]:
            for k in self.eligible[a]:
                if (a, k) not in self.Q:
                    raise ValueError(f"Missing Q[{a},{k}].")
                if (a, k) not in self.L:
                    raise ValueError(f"Missing L[{a},{k}].")
        for s in self.specialists:
            for k in self.eligible[s]:
                if (s, k) not in self.L:
                    raise ValueError(f"Missing L[{s},{k}].")
        cheapest = min(self.mu.values()) if self.mu else math.inf
        if cheapest > self.M:
            raise ValueError(
                f"Even the smallest config ({cheapest:.2f} GB) exceeds budget M={self.M} GB.")


@dataclass
class MamapSolution:
    status: str
    feasible: bool
    objective: float
    allocation: dict[str, str]
    loaded: list[str]
    per_agent_latency: dict[str, float]
    per_agent_energy: dict[str, float]
    router_quality: float = float("nan")
    synth_quality: float = float("nan")
    specialist_quality_gated: float = float("nan")
    system_latency: float = float("nan")

    @property
    def total_energy(self) -> float:
        return sum(v for v in self.per_agent_energy.values() if v == v)


def solve_mamap(inst: MamapInstance,
                eps: Optional[float] = None,
                solver: Optional[pulp.LpSolver] = None,
                msg: bool = False,
                *,
                latency_model: str = "worst_case",
                k_activated: int = 1,
                weights: Optional[tuple[float, float, float]] = None,
                normalize_specialists: bool = False) -> MamapSolution:
    """Solve the end-to-end MAMAP MILP.

    The defaults reproduce the v3 formulation exactly. Three latency models are
    available; the default ("worst_case") does not change prior behaviour:

    latency_model : {"worst_case", "sequential", "expected_max"}
        "worst_case" (default, v3 §9.1): L_max_spec >= L_s for EVERY specialist,
            so the SLO holds even if the router activates all of them. Because the
            specialists share ONE loaded model, this stage cost is a SINGLE
            specialist call (the max collapses to one), independent of k. This is
            the CONCURRENT / batched-stage model.
        "sequential": the paper's headline chain (§6.2, Eq. 7),
            L_router + k * L_spec + L_synth <= eps, where `k_activated` specialists
            run one-after-another on the shared instance. The specialist stage is
            multiplied by `k_activated`. Use this to reproduce the paper's
            capacity--latency frontier (Fig. 7 / Table 3), sweeping k in {1,3,5,9}.
        "expected_max": the per-query parallel-stage latency is the slowest
            ACTIVE specialist on that query, and the constraint bounds the MEAN
            over queries: (1/N) Σ_q max_{s∈A_q} L_s + L_router + L_synth <= eps.
            Requires `inst.per_query_active` (gold active sets A_q). This is the
            architecturally-correct "expected latency" for PARALLEL specialists;
            note the naive Σ_s π_s·L_s sketched in v3 §9.1 is a SERIAL-work proxy
            and is deliberately NOT implemented.

    k_activated : int
        Number of activated specialists on the critical path. ONLY used by the
        "sequential" model, where the specialist stage is k * L_spec (default 1).
        Ignored by "worst_case" and "expected_max".

    weights : (w_rt, w_syn, w_spec) or None
        Optional per-role weights on the three objective blocks (router,
        synthesiser, specialists). None (default) -> all 1.0 -> the raw v3 sum.

    normalize_specialists : bool
        If True, the specialist block is divided by the number of specialists so
        it lives on the same [0,1] scale as the router and synth terms, instead
        of summing up to ~|specialists| (Finding 4). Default False -> raw sum.
    """
    if latency_model not in ("worst_case", "sequential", "expected_max"):
        raise ValueError(f"unknown latency_model {latency_model!r}")
    if latency_model == "sequential" and (not isinstance(k_activated, int) or k_activated < 1):
        raise ValueError(f"latency_model='sequential' needs integer k_activated>=1, got {k_activated!r}")
    w_rt, w_syn, w_spec = weights if weights is not None else (1.0, 1.0, 1.0)
    inst.validate()
    prob = pulp.LpProblem("MAMAP_v2", pulp.LpMaximize)

    x = {(a, k): pulp.LpVariable(f"x_{a}_{k}", cat="Binary")
         for a in inst.agents for k in inst.eligible[a]}
    y = {k: pulp.LpVariable(f"y_{k}", cat="Binary") for k in inst.configs}

    # z aux for the bilinear gating terms; only created for triples with non-zero Q3
    # (the objective is max, so zero-Q3 z's would never be set anyway).
    z = {}
    for (s, k_s, k_d), q3 in inst.Q3.items():
        if s not in inst.specialists:
            continue
        if k_s not in inst.eligible.get(s, []):
            continue
        if k_d not in inst.eligible.get(inst.dispatcher, []):
            continue
        if not math.isfinite(q3) or q3 == 0.0:
            continue
        z[(s, k_s, k_d)] = pulp.LpVariable(
            f"z_{s}_{k_s}_{k_d}", lowBound=0.0, upBound=1.0, cat="Continuous")

    # objective: linear (dispatcher + synth) + gated (specialists).
    # Optional per-role weights + specialist normalisation (Finding 4); with the
    # defaults (weights=(1,1,1), normalize_specialists=False) this is the exact
    # raw v3 sum.
    router_terms, synth_terms = [], []
    for k in inst.eligible[inst.dispatcher]:
        q = inst.Q.get((inst.dispatcher, k), 0.0)
        if q and math.isfinite(q):
            router_terms.append(q * x[(inst.dispatcher, k)])
    for k in inst.eligible[inst.synthesiser]:
        q = inst.Q.get((inst.synthesiser, k), 0.0)
        if q and math.isfinite(q):
            synth_terms.append(q * x[(inst.synthesiser, k)])
    gated_terms = [inst.Q3[(s, k_s, k_d)] * z[(s, k_s, k_d)] for (s, k_s, k_d) in z]
    spec_scale = (1.0 / len(inst.specialists)
                  if (normalize_specialists and inst.specialists) else 1.0)
    prob += (w_rt * pulp.lpSum(router_terms)
             + w_syn * pulp.lpSum(synth_terms)
             + w_spec * spec_scale * pulp.lpSum(gated_terms)), "system_quality"

    # (4) one config per agent
    for a in inst.agents:
        prob += pulp.lpSum(x[(a, k)] for k in inst.eligible[a]) == 1, f"alloc_{a}"

    # (5) load-use
    for (a, k) in x:
        prob += x[(a, k)] <= y[k], f"loaduse_{a}_{k}"

    # (6) memory budget
    prob += pulp.lpSum(inst.mu[k] * y[k] for k in inst.configs) <= inst.M, "memory"

    # (6b) one context per (m,q)
    groups = inst.groups if inst.groups is not None else _auto_groups(inst.configs)
    for g, ks in groups.items():
        ks_in = [k for k in ks if k in y]
        if len(ks_in) > 1:
            prob += pulp.lpSum(y[k] for k in ks_in) <= 1, f"onectx_{g}"

    # z linearisation
    for (s, k_s, k_d), zv in z.items():
        prob += zv <= x[(s, k_s)],                          f"zub1_{s}_{k_s}_{k_d}"
        prob += zv <= x[(inst.dispatcher, k_d)],            f"zub2_{s}_{k_s}_{k_d}"
        prob += zv >= x[(s, k_s)] + x[(inst.dispatcher, k_d)] - 1, f"zlb_{s}_{k_s}_{k_d}"

    # (7) system-level latency: router + specialist-stage + synth <= eps.
    # The specialist stage runs in PARALLEL, so its wall-clock is a max over the
    # specialists that run. Two models for "which specialists run" (see audit):
    if eps is not None and math.isfinite(eps):
        L_router = pulp.lpSum(inst.L[(inst.dispatcher, k)] * x[(inst.dispatcher, k)]
                              for k in inst.eligible[inst.dispatcher])
        L_synth = pulp.lpSum(inst.L[(inst.synthesiser, k)] * x[(inst.synthesiser, k)]
                             for k in inst.eligible[inst.synthesiser])

        if latency_model == "worst_case":
            # v3 §9.1 default: max over ALL specialists (SLO holds for any routing).
            L_max_spec = pulp.LpVariable("L_max_spec", lowBound=0.0, cat="Continuous")
            for s in inst.specialists:
                prob += (L_max_spec >=
                         pulp.lpSum(inst.L[(s, k)] * x[(s, k)] for k in inst.eligible[s]),
                         f"lmaxspec_{s}")
            prob += L_router + L_max_spec + L_synth <= eps, "system_latency"

        elif latency_model == "sequential":
            # Paper §6.2 Eq. (7): the k activated specialists run one-after-another
            # on the shared instance, so the specialist stage is k * L_spec. The
            # specialists share one model (one specialist-slot config chosen for
            # all), so L_spec is that single shared config's latency; multiplying
            # by k_activated gives the sequential chain. We bound the slot's latency
            # by an auxiliary L_spec_one >= L_s for every specialist (they share the
            # config, so all these are equal at the optimum) and charge k * it.
            L_spec_one = pulp.LpVariable("L_spec_one", lowBound=0.0, cat="Continuous")
            for s in inst.specialists:
                prob += (L_spec_one >=
                         pulp.lpSum(inst.L[(s, k)] * x[(s, k)] for k in inst.eligible[s]),
                         f"lspecone_{s}")
            prob += L_router + k_activated * L_spec_one + L_synth <= eps, "system_latency"

        else:  # latency_model == "expected_max"
            # Architecturally-correct expected latency for PARALLEL specialists:
            # mean over queries of the slowest ACTIVE specialist on that query.
            # For each query q with gold active set A_q, L_q >= L_s for s in A_q;
            # constraint bounds (1/N) Σ_q L_q + L_router + L_synth.
            # (The naive Σ_s π_s·L_s of v3 §9.1 is a SERIAL-work proxy, not this.)
            if not inst.per_query_active:
                raise ValueError(
                    "latency_model='expected_max' requires inst.per_query_active "
                    "(per-query gold active specialist sets). Build it in derive.py "
                    "or use latency_model='worst_case'.")
            # only queries that actually activate >=1 specialist contribute a max term
            active_q = {q: [s for s in ss if s in inst.specialists]
                        for q, ss in inst.per_query_active.items()}
            active_q = {q: ss for q, ss in active_q.items() if ss}
            n_q = len(active_q)
            if n_q == 0:
                raise ValueError("per_query_active has no queries with specialists.")
            Lq_vars = {}
            for q, ss in active_q.items():
                Lq = pulp.LpVariable(f"Lq_{q}", lowBound=0.0, cat="Continuous")
                Lq_vars[q] = Lq
                for s in ss:
                    prob += (Lq >=
                             pulp.lpSum(inst.L[(s, k)] * x[(s, k)] for k in inst.eligible[s]),
                             f"lq_{q}_{s}")
            L_exp_spec = (1.0 / n_q) * pulp.lpSum(Lq_vars.values())
            prob += L_router + L_exp_spec + L_synth <= eps, "system_latency"

    prob.solve(solver or pulp.PULP_CBC_CMD(msg=msg))
    status = pulp.LpStatus[prob.status]
    feasible = status == "Optimal"

    if not feasible:
        return MamapSolution(status=status, feasible=False, objective=float("nan"),
                             allocation={}, loaded=[], per_agent_latency={},
                             per_agent_energy={})

    allocation = {a: k for (a, k) in x if x[(a, k)].value() and x[(a, k)].value() > 0.5}
    loaded = [k for k in inst.configs if y[k].value() and y[k].value() > 0.5]
    per_lat = {a: inst.L[(a, allocation[a])] for a in inst.agents}
    per_en = {a: inst.E.get((a, allocation[a]), float("nan")) for a in inst.agents}

    k_d = allocation[inst.dispatcher]
    k_synth = allocation[inst.synthesiser]
    router_q = inst.Q.get((inst.dispatcher, k_d), float("nan"))
    synth_q = inst.Q.get((inst.synthesiser, k_synth), float("nan"))
    spec_q = sum(inst.Q3.get((s, allocation[s], k_d), 0.0) for s in inst.specialists)
    # report system latency under the SAME model used to constrain it, so the
    # Pareto latency axis is consistent with the chosen latency_model.
    if latency_model == "expected_max" and inst.per_query_active:
        active_q = {q: [s for s in ss if s in inst.specialists]
                    for q, ss in inst.per_query_active.items()}
        active_q = {q: ss for q, ss in active_q.items() if ss}
        if active_q:
            spec_stage = sum(max(per_lat[s] for s in ss)
                             for ss in active_q.values()) / len(active_q)
        else:
            spec_stage = 0.0
    elif latency_model == "sequential":
        # k activated specialists run sequentially on the shared instance.
        spec_stage = k_activated * max(per_lat[s] for s in inst.specialists)
    else:
        spec_stage = max(per_lat[s] for s in inst.specialists)
    sys_lat = per_lat[inst.dispatcher] + spec_stage + per_lat[inst.synthesiser]

    return MamapSolution(
        status=status, feasible=True,
        objective=pulp.value(prob.objective),
        allocation=allocation, loaded=loaded,
        per_agent_latency=per_lat, per_agent_energy=per_en,
        router_quality=router_q, synth_quality=synth_q,
        specialist_quality_gated=spec_q, system_latency=sys_lat,
    )
