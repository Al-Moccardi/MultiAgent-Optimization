"""
part1_allocation/optimize/pareto.py
===================================
Capacity(=measured quality) vs latency Pareto front via the epsilon-constraint
method (formulazione section 7).

We sweep the per-agent latency cap `eps` over a grid, solve MAMAP-epsilon for
each, then drop dominated solutions comparing (total_quality [max], max_latency [min]).
"""
from __future__ import annotations

import math

from .milp import MamapInstance, MamapSolution, solve_mamap


def latency_grid(inst: MamapInstance, n: int = 12,
                 latency_model: str = "worst_case", k_activated: int = 1) -> list[float]:
    """Build a grid of epsilon thresholds on the SYSTEM-LATENCY scale.

    The latency constraint (milp.py, eq. 8) bounds the per-query inference CHAIN

        L_router + <specialist stage> + L_synth  <=  eps

    where the specialist stage depends on the latency model:
      * "worst_case"/"expected_max": a SINGLE specialist's latency (shared model),
      * "sequential" (paper Eq. 7): k_activated * a single specialist's latency.

    The grid lives on that same sum-of-three-roles scale. A grid built on a single
    agent's latency range is on the wrong scale: with specialists + synth present,
    every such eps sits BELOW the feasible system latency, so the whole sweep
    collapses onto the unconstrained point and the latency frontier is never traced.

    Bounds:
      lo : smallest FEASIBLE system latency (cheapest router + cheapest-of-slowest
           specialist [* k for sequential] + cheapest synth).
      hi : loosest system latency (slowest everything [* k for sequential]).
    """
    kmul = k_activated if latency_model == "sequential" else 1

    def _role_min(a: str) -> float:
        return min(inst.L[(a, k)] for k in inst.eligible[a])

    def _role_max(a: str) -> float:
        return max(inst.L[(a, k)] for k in inst.eligible[a])

    spec_min = max(_role_min(s) for s in inst.specialists)   # slowest unavoidable
    spec_max = max(_role_max(s) for s in inst.specialists)   # slowest possible
    lo = _role_min(inst.dispatcher) + kmul * spec_min + _role_min(inst.synthesiser)
    hi = _role_max(inst.dispatcher) + kmul * spec_max + _role_max(inst.synthesiser)
    if hi <= lo or n < 2:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def build_frontier(inst: MamapInstance, grid: list[float] | None = None,
                   n: int = 12, include_unconstrained: bool = True,
                   solve_kwargs: dict | None = None
                   ) -> list[tuple[float, MamapSolution]]:
    """Return non-dominated [(epsilon, solution)] sorted by latency ascending.

    `solve_kwargs` is forwarded verbatim to `solve_mamap` (e.g.
    {"latency_model": "expected_max"} or {"normalize_specialists": True}). Empty
    by default, so the sweep reproduces the v3 behaviour unchanged.
    """
    solve_kwargs = solve_kwargs or {}
    if grid is None:
        grid = latency_grid(inst, n=n,
                            latency_model=solve_kwargs.get("latency_model", "worst_case"),
                            k_activated=solve_kwargs.get("k_activated", 1))
    eps_values = list(grid)
    if include_unconstrained:
        eps_values.append(math.inf)

    raw: list[tuple[float, MamapSolution]] = []
    for eps in eps_values:
        sol = solve_mamap(inst, eps=eps, **solve_kwargs)
        if sol.feasible:
            raw.append((eps, sol))

    # Dominance: s1 dominates s2 if quality1 >= quality2 and maxlat1 <= maxlat2
    # with at least one strict. Maximize quality, minimize latency.
    kept: list[tuple[float, MamapSolution]] = []
    for eps, s in raw:
        dominated = False
        for _, t in raw:
            if t is s:
                continue
            if (t.objective >= s.objective - 1e-9
                    and t.system_latency <= s.system_latency + 1e-9
                    and (t.objective > s.objective + 1e-9
                         or t.system_latency < s.system_latency - 1e-9)):
                dominated = True
                break
        if not dominated:
            kept.append((eps, s))

    # De-duplicate identical (quality, latency) points, keep the tightest eps.
    seen: dict[tuple[int, int], tuple[float, MamapSolution]] = {}
    for eps, s in kept:
        key = (round(s.objective, 6), round(s.system_latency, 6))
        if key not in seen or eps < seen[key][0]:
            seen[key] = (eps, s)
    out = list(seen.values())
    out.sort(key=lambda es: es[1].system_latency)
    return out
