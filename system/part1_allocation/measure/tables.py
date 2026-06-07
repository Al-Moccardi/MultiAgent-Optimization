"""
part1_allocation/measure/tables.py
==================================
Read/write the two measurement tables and assemble the shared ParetoBundle.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from shared.schema import (AgentSpec, ConfigSpec, DeviceSpec, QualityRecord,
                           PerfRecord, LadderRung, ParetoSolution, ParetoBundle)
from part1_allocation.optimize.milp import MamapInstance, MamapSolution


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def quality_records_to_df(records: list[QualityRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def perf_records_to_df(records: list[PerfRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def save_df(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_df(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Bundle assembly
# --------------------------------------------------------------------------- #
def to_pareto_solution(eps: float, sol: MamapSolution, inst: MamapInstance
                       ) -> ParetoSolution:
    return ParetoSolution(
        epsilon_s=(float("inf") if eps == float("inf") else float(eps)),
        total_quality=float(sol.objective),
        max_latency_s=float(sol.system_latency),
        total_energy_j=float(sol.total_energy),
        allocation=dict(sol.allocation),
        loaded=list(sol.loaded),
        per_agent_latency_s=dict(sol.per_agent_latency),
        per_agent_energy_j=dict(sol.per_agent_energy),
    )


def build_ladders(inst: MamapInstance, configs: dict[str, ConfigSpec]
                  ) -> dict[str, list[LadderRung]]:
    """Per-agent ladder. For the dispatcher and synth, quality is linear (their
    own Q). For specialists, the bilinear Q3[s, k_a, k_d] depends also on the
    dispatcher's config; for the ladder we MARGINALISE over the dispatcher,
    taking the best k_d for each (s, k_a) -- i.e. the agent's upside quality
    given the freedom to pick the best router. This gives an interpretable
    cascade rung 'how good can this specialist config be at best'."""
    ladders: dict[str, list[LadderRung]] = {}
    for a in inst.agents:
        rungs = []
        for k in inst.eligible[a]:
            c = configs[k]
            if a == inst.dispatcher or a == inst.synthesiser:
                q = inst.Q.get((a, k), float("nan"))
            else:  # specialist
                q3 = [inst.Q3.get((a, k, k_d), 0.0)
                      for k_d in inst.eligible.get(inst.dispatcher, [])]
                q = max(q3) if q3 else float("nan")
            rungs.append(LadderRung(
                config_id=k, model=c.model, quant=c.quant, context=c.context,
                quality=q, latency_s=inst.L[(a, k)],
                energy_j=inst.E[(a, k)], peak_mem_gb=inst.mu[k],
            ))
        rungs.sort(key=lambda r: r.quality)
        ladders[a] = rungs
    return ladders


def assemble_bundle(frontier: list[tuple[float, MamapSolution]],
                    inst: MamapInstance,
                    configs: dict[str, ConfigSpec],
                    device: DeviceSpec,
                    hardware: str,
                    manifest: dict) -> ParetoBundle:
    return ParetoBundle(
        device=device,
        hardware=hardware,
        frontier=[to_pareto_solution(eps, sol, inst) for eps, sol in frontier],
        ladders=build_ladders(inst, configs),
        configs=configs,
        manifest=manifest,
    )
