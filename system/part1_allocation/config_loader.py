"""
part1_allocation/config_loader.py
=================================
Load YAML/JSONL configuration into typed objects.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from shared.schema import AgentSpec, ConfigSpec, DeviceSpec
from part1_allocation.scoring.scorer import Sample


def load_agents(path: str | Path) -> list[AgentSpec]:
    data = yaml.safe_load(Path(path).read_text())
    out = []
    for a in data["agents"]:
        out.append(AgentSpec(
            agent_id=a["id"], name=a.get("name", a["id"]),
            description=a.get("description", ""),
            c_min=int(a.get("c_min", 2048)),
            is_dispatcher=bool(a.get("is_dispatcher", False)),
        ))
    return out


def config_id(model_key: str, quant: str, context: int) -> str:
    return f"{model_key}__{quant}__c{context}"


def load_catalog(path: str | Path) -> dict[str, ConfigSpec]:
    data = yaml.safe_load(Path(path).read_text())
    configs: dict[str, ConfigSpec] = {}
    for m in data["models"]:
        key = m["key"]
        for q in m["quantizations"]:
            for ctx in m["contexts"]:
                cid = config_id(key, q["label"], ctx)
                configs[cid] = ConfigSpec(
                    config_id=cid, model=m.get("name", key), hf_id=m.get("hf_id", ""),
                    quant=q["label"], context=int(ctx),
                    gguf_path=q.get("gguf_path", "").format(context=ctx),
                    params_b=m.get("params_b"),
                )
    return configs


def load_device(path: str | Path) -> DeviceSpec:
    d = yaml.safe_load(Path(path).read_text())
    return DeviceSpec(
        name=d["name"], hw_class=d["hw_class"],
        memory_budget_gb=float(d["memory_budget_gb"]),
        latency_sla_s=float(d["latency_sla_s"]),
        latency_metric=d.get("latency_metric", "p95"),
        energy_backend=d.get("energy_backend", "none"),
    )


def load_testset(path: str | Path) -> dict[str, list[Sample]]:
    """JSONL with fields: query_id, agent, question, contexts[], ground_truth?"""
    by_agent: dict[str, list[Sample]] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        s = Sample(query_id=r["query_id"], agent=r["agent"], question=r["question"],
                   contexts=r.get("contexts", []), ground_truth=r.get("ground_truth"))
        by_agent.setdefault(s.agent, []).append(s)
    return by_agent


# --------------------------------------------------------------------------- #
# Calibration / routing gold set (calib_clean.yaml)
# --------------------------------------------------------------------------- #
def load_calib(path: str | Path) -> list[dict]:
    """Parse calib_clean.yaml -> list of raw query dicts."""
    data = yaml.safe_load(Path(path).read_text())
    return list(data.get("queries", []))


def calib_to_dispatcher_samples(calib: list[dict],
                                dispatcher_id: str = "A_dispatcher") -> list[Sample]:
    """Each calib query becomes ONE dispatcher sample whose gold label is the
    multi-label expected_agents set. Used to measure routing accuracy."""
    out = []
    for q in calib:
        out.append(Sample(
            query_id=q["id"], agent=dispatcher_id, question=q["text"],
            contexts=[], ground_truth=None,
            expected_agents=list(q.get("expected_agents", [])),
            difficulty=q.get("difficulty", ""), risk_level=q.get("risk_level", ""),
            category=q.get("category", ""), high_risk=bool(q.get("high_risk", False)),
        ))
    return out


def calib_to_specialist_samples(calib: list[dict],
                                dispatcher_id: str = "A_dispatcher") -> list[Sample]:
    """Fan out each calib query to each of its expected specialist agents, so the
    specialists are scored on the queries actually relevant to them, carrying the
    difficulty / risk labels. NOTE: these have no RAG contexts or answer ground
    truth -- supply a retriever / reference-free correctness judge for real runs."""
    out = []
    for q in calib:
        for a in q.get("expected_agents", []):
            if a == dispatcher_id:
                continue
            out.append(Sample(
                query_id=f"{q['id']}::{a}", agent=a, question=q["text"],
                contexts=[], ground_truth=None,
                expected_agents=list(q.get("expected_agents", [])),
                difficulty=q.get("difficulty", ""), risk_level=q.get("risk_level", ""),
                category=q.get("category", ""), high_risk=bool(q.get("high_risk", False)),
            ))
    return out


# --------------------------------------------------------------------------- #
# Gold set WITH ground-truth answers (calib_clean_with_gold.yaml) -> enables the
# specialist (RAGAS-style) evaluation. The dispatcher path is unchanged.
# --------------------------------------------------------------------------- #
def load_gold_calib(path: str | Path) -> list[dict]:
    """Parse calib_clean_with_gold.yaml -> list of raw query dicts (with
    ground_truth_answer, gold_law_ids, gold_case_ids)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return list(data.get("queries", []))


def gold_to_dispatcher_samples(gold: list[dict],
                               dispatcher_id: str = "A_dispatcher") -> list[Sample]:
    """Routing samples from the gold set (same as calib_to_dispatcher_samples)."""
    return calib_to_dispatcher_samples(gold, dispatcher_id)


def gold_to_specialist_samples(gold: list[dict], corpus=None,
                               dispatcher_id: str = "A_dispatcher") -> list[Sample]:
    """Fan out each gold query to its expected specialists, carrying the
    ground_truth_answer and the gold doc ids. If a CorpusStore is supplied, the gold
    ids are resolved to context TEXT (oracle-retrieval contexts) so the grounding
    metrics can run; without a corpus, contexts are empty (correctness-only)."""
    out = []
    for q in gold:
        law_ids = list(q.get("gold_law_ids", []))
        case_ids = list(q.get("gold_case_ids", []))
        contexts = corpus.contexts_for(law_ids, case_ids) if corpus is not None else []
        for a in q.get("expected_agents", []):
            if a == dispatcher_id:
                continue
            out.append(Sample(
                query_id=f"{q['id']}::{a}", agent=a, question=q["text"],
                contexts=contexts,
                ground_truth=q.get("ground_truth_answer"),
                expected_agents=list(q.get("expected_agents", [])),
                difficulty=q.get("difficulty", ""), risk_level=q.get("risk_level", ""),
                category=q.get("category", ""), high_risk=bool(q.get("high_risk", False)),
                gold_law_ids=law_ids, gold_case_ids=case_ids,
            ))
    return out
