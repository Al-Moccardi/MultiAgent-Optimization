"""
part4_dynamic_path/src/costq.py
===============================
Cost model and measured-quality tables for the dynamic layer, reusing the
artefacts produced by Parts 1-3 (no new measurement).

COST (per query) follows the paper's chain, with the executed specialist set
S(q). Under the shared-specialist abstraction the |S| activated specialists run
as ONE batched pass on the single loaded specialist instance, so the parallel
stage costs the latency of one specialist call (the concurrent / Lambda model
of section 10.5):

      L_sys(S)  =  L_disp  +  L_spec_one  +  L_synth          (concurrent)
      L_sys(S)  =  L_disp  +  |S| * L_spec_one  +  L_synth    (sequential, worst case)

Energy is additive over actually-executed agents:
      E_sys(S)  =  E_disp  +  |S| * E_spec_one  +  E_synth

Numbers come from the measured perf table (latency/energy per chosen config at
the fixed Part-1/2/3 allocation). Memory is fixed (models are pre-loaded), so
it does not enter the per-query decision.

QUALITY: Q_spec(M_s, domain) -- the measured per-(specialist model, domain)
quality from the quality table -- weights each candidate's value.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


# --------------------------------------------------------------------------- quality
def specialist_quality_by_domain(quality_table: str | Path,
                                 specialist_model: str) -> dict[str, float]:
    """Mean measured specialist quality per domain for the chosen specialist
    model (e.g. the Part-2 optimum 'llama3.2-1b'). Domain id = 'A_' + agent.

    Falls back to the pooled per-domain mean over ALL specialist configs if the
    specific model is sparse for a domain, so every domain gets a value.
    """
    qt = pd.read_parquet(quality_table)
    qt = qt[~qt.agent.isin(["A_dispatcher", "A_synth"])].dropna(subset=["quality"]).copy()
    qt["model"] = qt.config_id.apply(lambda c: str(c).split("__")[0])
    out: dict[str, float] = {}
    pooled = qt.groupby("agent").quality.mean()
    sub = qt[qt.model == specialist_model]
    bym = sub.groupby("agent").quality.mean() if len(sub) else pd.Series(dtype=float)
    for agent in qt.agent.unique():
        if agent in bym.index and not np.isnan(bym[agent]):
            out[agent] = float(bym[agent])
        else:
            out[agent] = float(pooled.get(agent, 0.6))
    return out


# --------------------------------------------------------------------------- cost
@dataclass
class CostModel:
    L_disp: float       # dispatcher latency (s) at the fixed allocation
    L_spec: float       # ONE specialist call latency (s)
    L_synth: float      # synthesiser latency (s)
    E_disp: float       # dispatcher energy (J)
    E_spec: float       # ONE specialist call energy (J)
    E_synth: float      # synthesiser energy (J)
    mode: str = "concurrent"   # {"concurrent", "sequential"}

    def latency(self, k: int) -> float:
        spec = self.L_spec if (self.mode == "concurrent") else k * self.L_spec
        # abstain (k=0): no specialist stage and no synthesis of specialists
        if k == 0:
            return self.L_disp
        return self.L_disp + spec + self.L_synth

    def energy(self, k: int) -> float:
        if k == 0:
            return self.E_disp
        return self.E_disp + k * self.E_spec + self.E_synth

    def spec_cost_one(self, metric: str = "latency") -> float:
        return self.L_spec if metric == "latency" else self.E_spec


def cost_model_from_perf(perf_table: str | Path,
                         disp_config: str, spec_config: str, synth_config: str,
                         mode: str = "concurrent") -> CostModel:
    """Build the cost model from the measured perf table for the three chosen
    role configs (the fixed Part-1/2/3 allocation). The perf table has derived
    latency/energy columns at the project's answer-length convention."""
    df = pd.read_parquet(perf_table)
    df = df.set_index("config_id") if "config_id" in df.columns else df
    # accept either 'ttft_s'+'throughput_tok_s' or precomputed latency/energy
    def lat(cfg, n):
        r = df.loc[cfg]
        if "latency_s" in df.columns:
            return float(r["latency_s"])
        ttft = float(r.get("ttft_s", 0.0))
        tput = float(r.get("throughput_tok_s", 1.0))
        return ttft + n / max(tput, 1e-6)
    def ene(cfg, n):
        r = df.loc[cfg]
        if "energy_j" in df.columns:
            return float(r["energy_j"])
        epb = float(r.get("energy_j_per_tok", 0.0))
        return epb * n
    N_R, N_G = 15, 384
    return CostModel(
        L_disp=lat(disp_config, N_R), L_spec=lat(spec_config, N_G),
        L_synth=lat(synth_config, N_G),
        E_disp=ene(disp_config, N_R), E_spec=ene(spec_config, N_G),
        E_synth=ene(synth_config, N_G), mode=mode)


# --------------------------------------------------------------------------- budget
def budget_from_entropy(signal_scores: dict[str, dict], cost_one: float,
                        k_min: int = 1, k_max: int = 6) -> float:
    """LLM-free, risk-free per-query budget: harder (more diffuse) queries get a
    bigger budget. Hardness = normalized entropy of the softmax over candidate
    fused scores; budget = cost of k_min..k_max specialist calls scaled by it.
    """
    vals = np.array([v["fused"] for v in signal_scores.values()], float)
    if vals.size == 0:
        return cost_one * k_min
    p = np.exp(vals - vals.max())
    p = p / p.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p) + 1e-12)  # in [0,1]
    k_budget = k_min + ent * (k_max - k_min)
    return float(cost_one * k_budget)
