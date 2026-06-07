"""
part1_allocation/factories.py
=============================
Build per-config backends and per-(config, agent) scorers, in either:
  * "mock" mode -- no models, deterministic, for plumbing/CI/dry runs.
  * "real" mode -- llama.cpp + RAGAS/Routing, for on-device measurement.

The scorer factory is AGENT-AWARE: the dispatcher gets a routing scorer
(quality = multi-label routing F1), specialists get RAGAS/correctness.

In mock mode we derive a per-config `quality_hint` from params_b so that bigger
models look better but cost more memory/latency -- giving a sensible synthetic
Pareto front (for both routing and specialist quality) to validate the optimizer.
"""
from __future__ import annotations

import math

from shared.schema import ConfigSpec, AgentSpec
from part1_allocation.inference.backend import (InferenceBackend, MockBackend,
                                                LlamaCppBackend)
from part1_allocation.scoring.scorer import (QualityScorer, MockScorer, RagasScorer,
                                             RoutingScorer, MockRoutingScorer,
                                             JudgeScorer, SpecialistScorer)


def _quality_hint(spec: ConfigSpec) -> float:
    """Map params + quant to a [0.2, 0.95] capability hint (mock only)."""
    p = spec.params_b or 1.0
    base = 0.30 + 0.55 * (math.log10(p + 0.5) - math.log10(0.5)) / (math.log10(8.0) - math.log10(0.5))
    if spec.quant.startswith("Q8") or spec.quant.upper() in ("F16", "FP16"):
        base += 0.03
    return max(0.2, min(0.95, base))


def make_mock_factories(agents: list[AgentSpec]):
    specialist_ids = [a.agent_id for a in agents if not a.is_dispatcher]

    def backend_factory(spec: ConfigSpec) -> InferenceBackend:
        return MockBackend(config_tag=spec.config_id, quality_hint=_quality_hint(spec),
                           params_b=spec.params_b, quant=spec.quant, context=spec.context)

    def scorer_factory(spec: ConfigSpec, agent: AgentSpec) -> QualityScorer:
        hint = _quality_hint(spec)
        if agent.is_dispatcher:
            return MockRoutingScorer(known_agents=specialist_ids, quality_hint=hint,
                                     config_tag=spec.config_id)
        return MockScorer(quality_hint=hint)

    return backend_factory, scorer_factory


def make_real_factories(agents: list[AgentSpec], n_gpu_layers: int = -1,
                        judge_llm=None, embeddings=None, correctness_fn=None,
                        judge_backend=None, embedder=None,
                        specialist_scorer: str = "judge",
                        corr_embedder=None, want_logprobs: bool = True):
    """Real llama.cpp + specialist scoring. Configure your judge / embedder.

    specialist_scorer:
      * "judge" (default) -> SpecialistScorer(judge_backend, embedder): correctness
        + grounding (if contexts) from the local judge, answer_relevancy + a
        semantic-correctness fallback from the embedder. Needs a judge and/or an
        embedder (at least one).
      * "ragas"           -> the literal RAGAS library via RagasScorer.

    want_logprobs:
      * True (default) -> backend runs with logits_all=True so token-level
        logprobs are available for the confidence signal (used only by the
        Stage-2 cascade).
      * False -> disables logits_all. This is 2-5x FASTER and uses less memory,
        at the cost of the confidence signal. For Stage-1 allocation (quality +
        latency + energy + memory) the confidence signal is unused, so this is
        the right setting for the measurement campaign.

    The dispatcher is always scored by RoutingScorer (multi-label routing F1).
    """
    specialist_ids = [a.agent_id for a in agents if not a.is_dispatcher]

    def backend_factory(spec: ConfigSpec) -> InferenceBackend:
        if not spec.gguf_path:
            raise ValueError(f"config {spec.config_id} has no gguf_path.")
        return LlamaCppBackend(gguf_path=spec.gguf_path, n_ctx=spec.context,
                               n_gpu_layers=n_gpu_layers,
                               want_logprobs=want_logprobs)

    def scorer_factory(spec: ConfigSpec, agent: AgentSpec) -> QualityScorer:
        if agent.is_dispatcher:
            return RoutingScorer(known_agents=specialist_ids)
        if specialist_scorer == "ragas":
            return RagasScorer(judge_llm=judge_llm, embeddings=embeddings,
                               correctness_fn=correctness_fn)
        if judge_backend is None and embedder is None:
            raise ValueError(
                "specialist scoring needs a judge and/or an embedder: pass "
                "--judge-gguf and/or --embedder, restrict to the router with "
                "--only-agents A_dispatcher, or use --specialist-scorer ragas.")
        # KEY POLICY:
        # The gold ground_truth_answer covers the FULL composed answer for each
        # query (often spanning multiple specialists' contributions). A single
        # specialist only addresses its domain slice; comparing its partial
        # output to the full ground truth would systematically penalise it.
        # So:
        #   * specialists -> include_correctness=False  (Q := faithfulness/relevancy mean)
        #   * synth       -> correctness_only=True       (Q := correctness alone)
        #     The synth is fed the real specialist answers and produces the final
        #     user-visible answer, so cos(answer, gold) already captures upstream
        #     errors; grounding would only dilute it.
        # q_correctness/faithfulness/relevancy are STILL measured & saved to
        # parquet for every agent; only their inclusion in the aggregate Q changes.
        is_synth = (agent.agent_id == "A_synth")
        return SpecialistScorer(judge_backend=judge_backend, embedder=embedder,
                                include_correctness=is_synth,
                                correctness_only=is_synth,
                                corr_embedder=corr_embedder)

    return backend_factory, scorer_factory
