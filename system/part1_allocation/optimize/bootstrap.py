"""
part1_allocation/optimize/bootstrap.py
======================================
Bootstrap confidence for the MAMAP optimum (Tier-3, Finding 8).

The quality coefficients Q3[s,k_a,k_d] are estimated from a small, imbalanced
gold set (25 queries; several specialists measured on <=3 queries). The argmax
allocation the MILP returns is therefore a point estimate over a noisy objective:
an allocation that wins by 0.005 in Q_sys may be statistically indistinguishable
from the runner-up. This module quantifies that uncertainty WITHOUT any new model
inference -- it resamples the already-measured per-query quality table.

Method (nonparametric bootstrap over queries):
  1. Take the measured quality_table + perf_table (the pipeline's own artefacts).
  2. For b in 1..B:
       - resample the ROOT query ids (the part of query_id before '::') with
         replacement, to the same count;
       - rebuild the quality rows for that resample (all agent/config rows whose
         root query id was drawn, with multiplicity);
       - rederive the MAMAP instance (build_instance) -> recomputes Q3, L, E, mu;
       - solve the MILP at the chosen eps / latency model / weights.
  3. Report, across the B solves:
       - objective mean and a percentile CI;
       - for each agent, how often each config_id was chosen (selection
         frequency) -- the practical "is this allocation robust?" signal;
       - the modal (most-frequent) full allocation and how often it won.

This is the analysis the optimisation deepdive recommended as priority #1. It is
pure post-processing on the parquet, so it runs in minutes and needs no GGUFs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Optional
import random

import pandas as pd

from shared.schema import AgentSpec, ConfigSpec, DeviceSpec
from .derive import build_instance
from .milp import solve_mamap


def _root_qid(qid: str) -> str:
    return str(qid).split("::", 1)[0]


def bootstrap_optimum(quality_df: pd.DataFrame,
                      perf_df: pd.DataFrame,
                      agents: Iterable[AgentSpec],
                      configs: dict[str, ConfigSpec],
                      device: DeviceSpec,
                      hardware: str,
                      *,
                      n_boot: int = 200,
                      eps: Optional[float] = None,
                      seed: int = 0,
                      dispatcher_id: str = "A_dispatcher",
                      synthesiser_id: str = "A_synth",
                      solve_kwargs: Optional[dict] = None,
                      ci: float = 0.95) -> dict:
    """Run the query-bootstrap and return a summary dict.

    Parameters mirror build_instance; `solve_kwargs` is forwarded to solve_mamap
    (e.g. {"latency_model": "expected_max"}). `eps` is the latency cap for every
    resample's solve (None -> unconstrained, matching a single-point study).
    """
    agents = list(agents)
    solve_kwargs = solve_kwargs or {}
    rng = random.Random(seed)

    qcol = quality_df.copy()
    qcol["_qid"] = qcol["query_id"].map(_root_qid)
    root_ids = sorted(qcol["_qid"].unique())
    n = len(root_ids)
    if n == 0:
        raise ValueError("no queries in quality_df to bootstrap.")

    # pre-group rows by root qid so each resample is a cheap concat
    rows_by_qid = {q: df.drop(columns="_qid") for q, df in qcol.groupby("_qid")}

    objectives: list[float] = []
    per_agent_choice: dict[str, Counter] = defaultdict(Counter)
    full_allocs: Counter = Counter()
    n_feasible = 0

    for _ in range(n_boot):
        draw = [root_ids[rng.randrange(n)] for _ in range(n)]
        # Concat with multiplicity. CRITICAL: derive.build_instance computes Q3 as
        # a MEAN over the ROOT query id (the part before "::"). So a repeated draw
        # must get a NEW ROOT id, otherwise duplicates collapse back into one group
        # and the resampling becomes a no-op. We therefore rewrite query_id to
        #   "<root>#b<j>[::<agent-suffix>]"
        # preserving any "::agent" tail so specialist/synth rows still parse.
        parts = []
        for j, q in enumerate(draw):
            sub = rows_by_qid[q].copy()
            new_root = f"{q}#b{j}"

            def _rewrite(qid: str, _nr=new_root) -> str:
                qid = str(qid)
                tail = qid.split("::", 1)
                return _nr if len(tail) == 1 else f"{_nr}::{tail[1]}"

            sub["query_id"] = sub["query_id"].map(_rewrite)
            parts.append(sub)
        resampled = pd.concat(parts, ignore_index=True)

        try:
            inst = build_instance(resampled, perf_df, agents, configs, device,
                                  hardware=hardware, dispatcher_id=dispatcher_id,
                                  synthesiser_id=synthesiser_id)
            sol = solve_mamap(inst, eps=eps, **solve_kwargs)
        except Exception:
            continue
        if not sol.feasible:
            continue
        n_feasible += 1
        objectives.append(float(sol.objective))
        for a, k in sol.allocation.items():
            per_agent_choice[a][k] += 1
        full_allocs[tuple(sorted(sol.allocation.items()))] += 1

    if n_feasible == 0:
        raise RuntimeError("no feasible bootstrap solves; check eps/budget.")

    objectives.sort()
    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    def _pct(p: float) -> float:
        i = min(len(objectives) - 1, max(0, int(round(p * (len(objectives) - 1)))))
        return objectives[i]

    # per-agent selection stability: fraction of resamples picking the modal config
    stability = {}
    for a, ctr in per_agent_choice.items():
        top_cfg, top_n = ctr.most_common(1)[0]
        stability[a] = {"modal_config": top_cfg,
                        "selection_freq": top_n / n_feasible,
                        "distinct_configs_seen": len(ctr)}

    modal_alloc, modal_n = full_allocs.most_common(1)[0]
    return {
        "n_boot": n_boot,
        "n_feasible": n_feasible,
        "objective_mean": sum(objectives) / len(objectives),
        "objective_ci": [_pct(lo_q), _pct(hi_q)],
        "objective_min": objectives[0],
        "objective_max": objectives[-1],
        "per_agent_stability": stability,
        "modal_allocation": dict(modal_alloc),
        "modal_allocation_freq": modal_n / n_feasible,
        "ci_level": ci,
    }


def print_bootstrap_summary(summary: dict) -> None:
    s = summary
    print(f"[bootstrap] {s['n_feasible']}/{s['n_boot']} feasible resamples")
    lo, hi = s["objective_ci"]
    print(f"[bootstrap] objective: mean={s['objective_mean']:.4f}  "
          f"{int(s['ci_level']*100)}% CI=[{lo:.4f}, {hi:.4f}]  "
          f"range=[{s['objective_min']:.4f}, {s['objective_max']:.4f}]")
    print(f"[bootstrap] modal allocation won {s['modal_allocation_freq']*100:.0f}% "
          f"of resamples")
    print("[bootstrap] per-agent selection stability (modal config / how often chosen):")
    for a, st in sorted(s["per_agent_stability"].items()):
        flag = "  <-- UNSTABLE" if st["selection_freq"] < 0.6 else ""
        print(f"    {a:48s} {st['modal_config']:28s} "
              f"{st['selection_freq']*100:3.0f}%  "
              f"({st['distinct_configs_seen']} configs seen){flag}")
