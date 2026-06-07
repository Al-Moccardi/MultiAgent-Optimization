"""
shared/schema.py
================
Single source of truth for the data shapes exchanged in the project.

This module is deliberately dependency-light (only stdlib) so that BOTH
`part1_allocation` (which *produces* the artifacts) and `part2_cascade`
(which *consumes* them) can import it without pulling in solver/inference
dependencies.

The interface contract between the two parts is the `shared/pareto/`
directory, whose contents are described by `ParetoBundle.FILES`.

Design decisions baked into these shapes (see formulazione / chat handoff):

* Quality is HARDWARE-INVARIANT  -> measured once, keyed by (agent, query, config).
* Performance is HARDWARE-SPECIFIC -> measured per hardware, keyed by (hardware, config).
* Per-query latency / energy are DERIVED, not separately measured:
      L = ttft + n_tokens / throughput
      E = n_tokens * energy_per_token
* The objective coefficient is MEASURED quality Q_{a,k} (no params proxy).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# Static specifications (inputs that describe the problem instance)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentSpec:
    """One agent of the multi-agent system. The dispatcher is an agent too."""
    agent_id: str
    name: str
    description: str
    c_min: int = 2048          # c°_a : minimum context length the agent requires
    is_dispatcher: bool = False


@dataclass(frozen=True)
class ConfigSpec:
    """A candidate configuration k = (model, quant, context)."""
    config_id: str             # canonical id, e.g. "qwen2.5-3b__Q4_K_M__c8192"
    model: str                 # human-readable model name
    hf_id: str                 # HuggingFace id (provenance only; NOT used as proxy)
    quant: str                 # quantization label, e.g. "Q4_K_M"
    context: int               # c_k : context length
    gguf_path: str             # local path to the GGUF artifact (llama.cpp)
    params_b: Optional[float] = None   # billions of params (provenance / tie-break only)


@dataclass(frozen=True)
class DeviceSpec:
    """A target device (one of the laptop classes we generalize across)."""
    name: str
    hw_class: str              # "UNIFIED" | "DISCRETE_GPU" | "CPU_ONLY"
    memory_budget_gb: float    # M : single memory-class budget (VRAM/unified/RAM)
    latency_sla_s: float       # T° : per-agent latency budget per call
    latency_metric: str = "p95"   # how to aggregate per-query latency to per-(a,k): "mean"|"p95"
    energy_backend: str = "none"  # "powermetrics"|"rapl"|"nvidia_smi"|"tegrastats"|"none"


# --------------------------------------------------------------------------- #
# Measurement records (the two tables)
# --------------------------------------------------------------------------- #
@dataclass
class QualityRecord:
    """One (agent, query, config) measurement. HARDWARE-INVARIANT."""
    agent: str
    query_id: str
    config_id: str
    output: str
    n_out_tokens: int
    # quality sub-scores in [0,1]; any may be NaN/absent depending on scorer
    q_faithfulness: float = float("nan")
    q_answer_relevancy: float = float("nan")
    q_context_precision: float = float("nan")
    q_context_recall: float = float("nan")
    q_correctness: float = float("nan")
    quality: float = float("nan")     # scalar aggregate used by the MILP objective
    confidence: float = float("nan")  # in [0,1], from logprob/perplexity (for the cascade)
    # query metadata (from the calibration/routing set) -- carried through for
    # difficulty/risk breakdowns and for Part 2's cascade & criticality policy.
    difficulty: str = ""              # "simple" | "medium" | "complex"
    risk_level: str = ""              # "low" | "medium" | "high"
    category: str = ""
    expected_agents: str = ""         # "|"-joined gold routing (dispatcher rows)


@dataclass
class PerfRecord:
    """One (hardware, config) measurement. HARDWARE-SPECIFIC."""
    hardware: str
    config_id: str
    peak_mem_gb: float         # mu_k on this hardware
    ttft_s: float              # ell_k
    throughput_tok_s: float    # tau_k
    energy_j_per_tok: float = float("nan")


# --------------------------------------------------------------------------- #
# Optimizer outputs (the SHARED CONTRACT with Part 2)
# --------------------------------------------------------------------------- #
@dataclass
class LadderRung:
    """A deployable config available to an agent, with derived costs.
    Part 2's cascade escalates *up* this ladder (sorted by quality asc->desc)."""
    config_id: str
    model: str
    quant: str
    context: int
    quality: float
    latency_s: float
    energy_j: float
    peak_mem_gb: float


@dataclass
class ParetoSolution:
    """One Pareto-efficient static allocation (output of MAMAP-epsilon)."""
    epsilon_s: float                      # latency cap used for this solve (inf -> unconstrained)
    total_quality: float                  # objective value P(x) = sum Q_{a,k} x_{a,k}
    max_latency_s: float                  # max_a L_a(s)  (the front's latency axis)
    total_energy_j: float                 # sum over agents of per-call energy
    allocation: dict[str, str]            # agent_id -> config_id  (the x solution)
    loaded: list[str]                     # config_ids with y_k = 1
    per_agent_latency_s: dict[str, float]
    per_agent_energy_j: dict[str, float]


@dataclass
class ParetoBundle:
    """Everything Part 1 hands to Part 2. Serialized into `shared/pareto/`."""
    device: DeviceSpec
    hardware: str
    frontier: list[ParetoSolution]
    # per-agent ladder restricted to configs loadable within budget, quality-sorted
    ladders: dict[str, list[LadderRung]]
    configs: dict[str, ConfigSpec]
    manifest: dict

    FILES = {
        "frontier":      "frontier.json",
        "ladders":       "ladders.json",
        "configs":       "configs.json",
        "manifest":      "manifest.json",
        # raw tables copied/linked so Part 2 can REPLAY the cascade offline:
        "quality_table": "quality_table.parquet",
        "perf_table":    "perf_table.parquet",
    }

    # -- serialization -----------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _dump(out / self.FILES["frontier"], [asdict(s) for s in self.frontier])
        _dump(out / self.FILES["ladders"],
              {a: [asdict(r) for r in rungs] for a, rungs in self.ladders.items()})
        _dump(out / self.FILES["configs"],
              {cid: asdict(c) for cid, c in self.configs.items()})
        _dump(out / self.FILES["manifest"], self.manifest | {"device": asdict(self.device),
                                                             "hardware": self.hardware})
        return out

    @staticmethod
    def load(in_dir: str | Path) -> "ParetoBundle":
        d = Path(in_dir)
        frontier = [ParetoSolution(**s) for s in _load(d / ParetoBundle.FILES["frontier"])]
        ladders = {a: [LadderRung(**r) for r in rungs]
                   for a, rungs in _load(d / ParetoBundle.FILES["ladders"]).items()}
        configs = {cid: ConfigSpec(**c) for cid, c in _load(d / ParetoBundle.FILES["configs"]).items()}
        man = _load(d / ParetoBundle.FILES["manifest"])
        device = DeviceSpec(**man.pop("device"))
        hardware = man.pop("hardware")
        return ParetoBundle(device=device, hardware=hardware, frontier=frontier,
                            ladders=ladders, configs=configs, manifest=man)


def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _load(path: Path):
    return json.loads(Path(path).read_text())
