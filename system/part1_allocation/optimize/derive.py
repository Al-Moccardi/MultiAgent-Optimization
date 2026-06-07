"""
part1_allocation/optimize/derive.py
====================================
Build the end-to-end MAMAP instance from the two measurement parquets.

Given the per-query quality table and per-config performance table, this:
  * detects roles  (dispatcher / specialists / synthesiser) from the agents list
  * builds linear Q[agent, k] for dispatcher and synth   (their own quality)
  * builds bilinear Q3[specialist, k_a, k_d]              (router-gated)
       Q3[s, k_a, k_d] = (1/N) * sum_{q in expected(s)} 1[s in pred(k_d, q)] * Qgen(s, k_a, q)
       where N = |expected(s)|, the queries on which we measured s.
  * carries L, E, mu through unchanged

The bilinear Q3 is computed offline from the parquets that the pipeline already
writes -- NO new model inference required to switch to the end-to-end objective.
"""
from __future__ import annotations

from typing import Iterable
import math

import pandas as pd

from .milp import MamapInstance, _auto_groups
from shared.schema import AgentSpec, ConfigSpec, DeviceSpec


def _agg(xs, how: str) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if how == "p95":
        i = max(0, int(0.95 * (len(xs) - 1)))
        return float(xs[i])
    return float(sum(xs) / len(xs))


def _parse_predicted(out: str) -> set[str]:
    """Dispatcher rows store predicted agent ids in `output` as 'A1|A2|...'.
    Empty/NaN -> empty set (router predicted nothing)."""
    if out is None or (isinstance(out, float) and math.isnan(out)):
        return set()
    s = str(out).strip()
    if not s:
        return set()
    return {tok for tok in s.split("|") if tok}


def build_instance(quality_df: pd.DataFrame,
                   perf_df: pd.DataFrame,
                   agents: Iterable[AgentSpec],
                   configs: dict[str, ConfigSpec],
                   device: DeviceSpec,
                   hardware: str,
                   *,
                   dispatcher_id: str = "A_dispatcher",
                   synthesiser_id: str = "A_synth") -> MamapInstance:
    """Build the end-to-end instance for one hardware target."""
    agents = list(agents)
    agent_ids = [a.agent_id for a in agents]
    if dispatcher_id not in agent_ids:
        raise ValueError(f"dispatcher '{dispatcher_id}' missing from agents list")
    if synthesiser_id not in agent_ids:
        raise ValueError(
            f"synthesiser '{synthesiser_id}' missing from agents list -- add it "
            f"to agents.yaml; under the end-to-end formulation every system must "
            f"have a designated synth agent producing the final user-visible answer.")
    spec_agents = [a for a in agents if a.agent_id not in (dispatcher_id, synthesiser_id)]

    perf_h = perf_df[perf_df["hardware"] == hardware].set_index("config_id")
    config_ids = list(configs)
    mu = {cid: float(perf_h.loc[cid, "peak_mem_gb"]) for cid in config_ids
          if cid in perf_h.index}

    # context-fit eligibility for every agent
    eligible: dict[str, list[str]] = {}
    for a in agents:
        eligible[a.agent_id] = [cid for cid in config_ids
                                if configs[cid].context >= a.c_min]

    Q: dict[tuple[str, str], float] = {}
    Q3: dict[tuple[str, str, str], float] = {}
    L: dict[tuple[str, str], float] = {}
    E: dict[tuple[str, str], float] = {}
    how = device.latency_metric

    # ---- linear quality + L,E for every agent ----
    for a in agents:
        sub_a = quality_df[quality_df["agent"] == a.agent_id]
        kept: list[str] = []
        for cid in eligible[a.agent_id]:
            rows = sub_a[sub_a["config_id"] == cid]
            if rows.empty or cid not in perf_h.index:
                continue
            ttft = float(perf_h.loc[cid, "ttft_s"])
            tput = float(perf_h.loc[cid, "throughput_tok_s"])
            ept = float(perf_h.loc[cid, "energy_j_per_tok"])
            q = float(rows["quality"].mean())
            if not math.isfinite(q):
                print(f"[derive] drop ({a.agent_id},{cid}): quality NaN")
                continue
            if not (math.isfinite(ttft) and math.isfinite(tput) and tput > 0):
                print(f"[derive] drop ({a.agent_id},{cid}): latency undefined")
                continue
            lat_q = [ttft + n / tput for n in rows["n_out_tokens"]]
            en_q = [n * ept for n in rows["n_out_tokens"]]
            Q[(a.agent_id, cid)] = q
            L[(a.agent_id, cid)] = _agg(lat_q, how)
            E[(a.agent_id, cid)] = _agg(en_q, "mean")
            kept.append(cid)
        eligible[a.agent_id] = kept
        if not kept:
            raise ValueError(
                f"Agent '{a.agent_id}' has no usable config after the quality/latency join.")

    # ---- bilinear Q3 for specialists, derived from per-query data ----
    # router predictions: query_id -> {k_d: set(predicted specialists)}
    disp_rows = quality_df[quality_df["agent"] == dispatcher_id]
    # which queries each specialist was actually evaluated on (from expected_agents fanout)
    # Each row's query_id is e.g. "Q05::A_specialist_X"; strip the agent suffix.
    def _root_qid(qid: str) -> str:
        return str(qid).split("::", 1)[0]

    pred_by_kd_query: dict[str, dict[str, set[str]]] = {}
    for _, r in disp_rows.iterrows():
        k_d = r["config_id"]
        qid = _root_qid(r["query_id"])
        pred = _parse_predicted(r.get("output"))
        pred_by_kd_query.setdefault(k_d, {})[qid] = pred

    for s in spec_agents:
        sid = s.agent_id
        spec_rows = quality_df[quality_df["agent"] == sid]
        if spec_rows.empty:
            continue
        # group per-query quality of this specialist by k_a (using mean over duplicates)
        spec_rows = spec_rows.assign(_qid=spec_rows["query_id"].map(_root_qid))
        # set of queries on which s was measured (= queries where s was expected)
        N_s = spec_rows["_qid"].nunique()
        if N_s == 0:
            continue
        # per (k_a, query) -> generation quality
        qgen = (spec_rows.groupby(["config_id", "_qid"])["quality"]
                         .mean().to_dict())  # (k_a, qid) -> Qgen
        for k_a in eligible[sid]:
            for k_d, pred_map in pred_by_kd_query.items():
                if k_d not in eligible[dispatcher_id]:
                    continue
                acc = 0.0
                for qid, pred in pred_map.items():
                    if sid in pred and (k_a, qid) in qgen:
                        v = qgen[(k_a, qid)]
                        if math.isfinite(v):
                            acc += v
                Q3[(sid, k_a, k_d)] = acc / N_s

    groups = _auto_groups(config_ids)

    # Per-query gold active specialist sets A_q (qid -> [specialist_id, ...]),
    # recovered from the dispatcher rows' expected_agents. OPTIONAL: only the
    # `expected_max` latency model consumes this; the default worst_case ignores it.
    per_query_active: dict[str, list[str]] = {}
    spec_ids = {s.agent_id for s in spec_agents}
    for _, r in disp_rows.iterrows():
        qid = _root_qid(r["query_id"])
        exp = r.get("expected_agents")
        if exp is None or (isinstance(exp, float) and math.isnan(exp)):
            continue
        ss = [a for a in str(exp).split("|") if a in spec_ids]
        if ss:
            per_query_active.setdefault(qid, [])
            for a in ss:
                if a not in per_query_active[qid]:
                    per_query_active[qid].append(a)

    return MamapInstance(
        specialists=[s.agent_id for s in spec_agents],
        dispatcher=dispatcher_id,
        synthesiser=synthesiser_id,
        configs=config_ids,
        eligible=eligible,
        Q=Q, Q3=Q3, L=L, E=E, mu=mu, M=device.memory_budget_gb,
        groups=groups,
        per_query_active=(per_query_active or None),
    )
