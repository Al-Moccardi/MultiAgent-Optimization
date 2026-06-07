"""
part1_allocation/measure/quality.py
===================================
Quality sweep (EXPENSIVE, run ONCE -- hardware-invariant).

Agent-aware scoring:
  * dispatcher (is_dispatcher=True) -> ROUTING prompt + routing scorer; quality
    is multi-label routing F1 (the MILP objective for the dispatcher).
  * specialists                     -> RAG prompt + RAGAS/correctness scorer.

QUALITY DEDUP (dedup_by_model_quant=True, default):
  A model's output is the same regardless of the *context-length* config variant
  (the routing prompt is tiny; for specialists, as long as the input fits the
  chosen context). So we measure quality ONCE per (model, quant) -- loading the
  model once -- and copy the measured record to every context-eligible config_id
  in that group. For a catalog with C context lengths this cuts the expensive
  sweep ~C-fold. Set dedup_by_model_quant=False to measure every config_id.

We log PER-(agent, query, config) records (incl. difficulty/risk and, for the
dispatcher, predicted vs expected agents) so Part 2 can replay offline.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from shared.schema import AgentSpec, ConfigSpec, QualityRecord
from part1_allocation.inference.backend import InferenceBackend
from part1_allocation.scoring.scorer import QualityScorer, Sample


def _root_qid(qid: str) -> str:
    """Strip the agent/router suffix: 'GS001::A_synth' -> 'GS001'."""
    return str(qid).split("::", 1)[0]

BackendFactory = Callable[[ConfigSpec], InferenceBackend]
ScorerFactory = Callable[[ConfigSpec, AgentSpec], QualityScorer]
PromptFn = Callable[[Sample], str]


def default_prompt_fn(sample: Sample) -> str:
    ctx = "\n\n".join(f"[CTX {i}] {c}" for i, c in enumerate(sample.contexts)) \
        if sample.contexts else "(nessun contesto recuperato)"
    return (f"Contesto:\n{ctx}\n\nDomanda: {sample.question}\n\n"
            f"Rispondi in italiano, citando gli articoli pertinenti.")


def routing_prompt(sample: Sample, specialists: list[AgentSpec]) -> str:
    catalog = "\n".join(f"- {a.agent_id}: {a.name}" for a in specialists)
    return (f"Sei un router per un assistente legale di diritto di famiglia.\n"
            f"Agenti disponibili:\n{catalog}\n\n"
            f"Domanda dell'utente: {sample.question}\n\n"
            f"Elenca gli id degli agenti competenti (anche piu' di uno), "
            f"scrivendo gli id esattamente come sopra.")


def _quality_key(config_id: str) -> str:
    """Group key = config_id without the trailing '__c<context>' part."""
    return config_id.rsplit("__", 1)[0]


def _cap_tokens(text: str, max_tok: int) -> str:
    """Roughly cap `text` to `max_tok` tokens (~chars/3.5 for Italian), on a word
    boundary. Used to give every activated specialist an EQUAL share of the
    synthesiser's input budget so no single specialist dominates."""
    if max_tok <= 0 or not text:
        return ""
    max_chars = int(max_tok * 3.5)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else cut).rstrip() + " ..."


def _compose_synth_input(question: str,
                         activated: list[str],
                         spec_answers: dict[str, str],
                         budget_tok: int) -> str:
    """Build the synthesiser's input from the REAL answers of the activated
    specialists, each capped to an equal share of `budget_tok`. This replaces the
    Mode-(a) approximation (idealised gold contexts) with the actual evidence the
    synth receives in deployment: the specialists' generated answers."""
    present = [s for s in activated if spec_answers.get(s)]
    if not present:
        return ""  # router activated nothing usable -> synth has no agent input
    share = max(1, budget_tok // len(present))
    parts = []
    for s in present:
        parts.append(f"[{s}]\n{_cap_tokens(spec_answers[s], share)}")
    return "\n\n".join(parts)


def measure_quality(agents: list[AgentSpec],
                    configs: dict[str, ConfigSpec],
                    samples_by_agent: dict[str, list[Sample]],
                    backend_factory: BackendFactory,
                    scorer_factory: ScorerFactory,
                    prompt_fn: PromptFn = default_prompt_fn,
                    max_tokens: int = 512,
                    routing_specialists: list[AgentSpec] | None = None,
                    dedup_by_model_quant: bool = True,
                    synth_id: str = "A_synth",
                    synth_input_budget: int | None = None,
                    checkpoint_path=None) -> list[QualityRecord]:
    """Quality sweep (two-pass; the synth is handled separately).

    Pass 1 measures every agent EXCEPT the synth, exactly as before, and records
    each specialist's answer at a fixed REFERENCE config (the largest-context
    group) for the synth. Pass 2 feeds the synth the COMPOSED real answers of the
    GOLD-activated specialists, each capped to an equal share of
    `synth_input_budget`, and scores it against the gold answer (one record per
    synth config; the MILP stays linear).

    Robustness for long real campaigns:
      * checkpoint_path : if given, the accumulated records are written to this
        path (parquet) after EVERY (model,quant) group, so a later crash does not
        lose completed work. Combined with run_all's --reuse-quality, a crashed
        run can be resumed.
      * a model that fails to load OR raises mid-generation (e.g. a CUDA OOM on an
        oversize config) is SKIPPED with a warning instead of aborting the whole
        sweep.
    """
    import pandas as pd  # local: only needed if checkpointing

    def _checkpoint():
        if checkpoint_path and records:
            try:
                pd.DataFrame([r.__dict__ for r in records]).to_parquet(checkpoint_path)
            except Exception as e:
                print(f"[quality] checkpoint write failed ({e}) -- continuing")

    records: list[QualityRecord] = []
    specialists = routing_specialists if routing_specialists is not None \
        else [a for a in agents if not a.is_dispatcher]
    synth_input_budget = synth_input_budget or max_tokens

    dispatcher = next((a for a in agents if a.is_dispatcher), None)
    synth_agent = next((a for a in agents if a.agent_id == synth_id), None)
    pass1_agents = [a for a in agents if a.agent_id != synth_id]

    # Group configs by (model, quant); each group's context variants share output.
    groups: dict[str, list[tuple[str, ConfigSpec]]] = defaultdict(list)
    for cid, spec in configs.items():
        key = _quality_key(cid) if dedup_by_model_quant else cid
        groups[key].append((cid, spec))

    # --- data recorded during pass 1 for the synth in pass 2 -----------------
    # reference specialist answers: pick the group with the LARGEST context as the
    # fixed reference (most capable / longest-context), same for all -> no
    # circularity with the MILP's allocation choice.
    ref_key = max(groups, key=lambda k: max(c.context for (_, c) in groups[k]))
    spec_ref_answer: dict[str, dict[str, str]] = defaultdict(dict)  # qid -> {spec_id: answer}

    def _emit(agent, sample, output, res, qs, fitting):
        for (cid, _c) in fitting:
            records.append(QualityRecord(
                agent=agent.agent_id, query_id=sample.query_id,
                config_id=cid, output=output, n_out_tokens=res.n_out_tokens,
                q_faithfulness=qs.faithfulness, q_answer_relevancy=qs.answer_relevancy,
                q_context_precision=qs.context_precision,
                q_context_recall=qs.context_recall, q_correctness=qs.correctness,
                quality=qs.quality,
                confidence=(res.confidence if res.confidence is not None else float("nan")),
                difficulty=sample.difficulty, risk_level=sample.risk_level,
                category=sample.category,
                expected_agents="|".join(sample.expected_agents),
            ))

    def _fitting(prompt, eligible_members):
        est = int(len(prompt) / 3.5) + max_tokens
        fit = [(cid, c) for (cid, c) in eligible_members if c.context >= est]
        return fit or [min(eligible_members, key=lambda cc: cc[1].context)]

    # ===================== PASS 1: everyone except the synth =================
    for key, members in groups.items():
        members_sorted = sorted(members, key=lambda cs: cs[1].context)
        rep_cid, rep_spec = members_sorted[-1]
        serving = [a for a in pass1_agents
                   if any(c.context >= a.c_min for (_, c) in members)]
        if not serving:
            continue
        try:
            backend = backend_factory(rep_spec)
        except Exception as e:
            print(f"[skip] quality '{rep_cid}': model failed to load "
                  f"({type(e).__name__}: {e})")
            continue
        try:
            for agent in serving:
                scorer = scorer_factory(rep_spec, agent)
                eligible_members = [(cid, c) for (cid, c) in members
                                    if c.context >= agent.c_min]
                for sample in samples_by_agent.get(agent.agent_id, []):
                    if agent.is_dispatcher:
                        prompt = routing_prompt(sample, specialists)
                    else:
                        prompt = prompt_fn(sample)
                    res = backend.generate(prompt, max_tokens=max_tokens,
                                           context=rep_spec.context)
                    qs = scorer.score(sample, res.text)
                    output = res.text
                    if qs.predicted_agents is not None:
                        output = "|".join(qs.predicted_agents)

                    # record this specialist's REFERENCE answer (from ref group only)
                    if (not agent.is_dispatcher) and key == ref_key:
                        spec_ref_answer[_root_qid(sample.query_id)][agent.agent_id] = res.text

                    _emit(agent, sample, output, res, qs, _fitting(prompt, eligible_members))
        except Exception as e:
            # e.g. a CUDA OOM mid-generation on an oversize config: skip this group
            # rather than aborting the whole (multi-hour) sweep. Records already
            # accumulated from earlier groups are preserved and checkpointed.
            print(f"[skip] quality '{rep_cid}': generation failed mid-group "
                  f"({type(e).__name__}: {e}) -- skipping the rest of this group")
        finally:
            backend.close()
        _checkpoint()  # persist after every (model,quant) group

    # ===================== PASS 2: the synthesiser ===========================
    # Path 2: the synth is fed the COMPOSED REAL answers of the GOLD-activated
    # specialists (each capped to an equal share of the budget), scored against
    # the gold answer. One record per synth config (MILP stays linear); the
    # router-config dependence is intentionally not modelled here.
    if synth_agent is not None and samples_by_agent.get(synth_id):
        synth_groups = {k: m for k, m in groups.items()
                        if any(c.context >= synth_agent.c_min for (_, c) in m)}
        for key, members in synth_groups.items():
            members_sorted = sorted(members, key=lambda cs: cs[1].context)
            rep_cid, rep_spec = members_sorted[-1]
            eligible_members = [(cid, c) for (cid, c) in members
                                if c.context >= synth_agent.c_min]
            try:
                backend = backend_factory(rep_spec)
            except Exception as e:
                print(f"[skip] quality synth '{rep_cid}': failed to load ({e})")
                continue
            try:
                scorer = scorer_factory(rep_spec, synth_agent)
                for sample in samples_by_agent[synth_id]:
                    root = _root_qid(sample.query_id)
                    answers = spec_ref_answer.get(root, {})
                    activated = [a for a in sample.expected_agents
                                 if a != (dispatcher.agent_id if dispatcher else "")]
                    composed = _compose_synth_input(
                        sample.question, activated, answers, synth_input_budget)
                    syn_sample = Sample(
                        query_id=sample.query_id, agent=synth_id,
                        question=sample.question,
                        contexts=([composed] if composed else sample.contexts),
                        ground_truth=sample.ground_truth,
                        expected_agents=sample.expected_agents,
                        difficulty=sample.difficulty, risk_level=sample.risk_level,
                        category=sample.category, high_risk=sample.high_risk,
                        gold_law_ids=sample.gold_law_ids,
                        gold_case_ids=sample.gold_case_ids,
                        retrieved_ids=sample.retrieved_ids)
                    prompt = prompt_fn(syn_sample)
                    res = backend.generate(prompt, max_tokens=max_tokens,
                                           context=rep_spec.context)
                    qs = scorer.score(syn_sample, res.text)
                    _emit(synth_agent, syn_sample, res.text, res, qs,
                          _fitting(prompt, eligible_members))
            except Exception as e:
                print(f"[skip] quality synth '{rep_cid}': generation failed "
                      f"({type(e).__name__}: {e}) -- skipping rest of this group")
            finally:
                backend.close()
            _checkpoint()
    return records
