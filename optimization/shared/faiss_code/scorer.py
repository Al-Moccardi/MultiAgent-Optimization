"""
part1_allocation/scoring/scorer.py
==================================
Quality scoring. The MILP objective is the scalar `quality` in [0,1].

Two implementations:
  * RagasScorer : faithfulness / answer-relevancy / context-precision / -recall
                  (+ optional correctness via a judge). Needs a judge LLM.
  * MockScorer  : deterministic, derives a plausible quality from the backend's
                  confidence and a per-config quality_hint (for dry runs).

IMPORTANT (from the chat): RAGAS measures GROUNDING, not legal CORRECTNESS.
The default aggregate therefore *down-weights* RAGAS-only signals and reserves
headroom for a correctness term; supply `q_correctness` from an expert-checked
subset or a correctness judge for the real runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math


@dataclass
class Sample:
    """One evaluation item routed to an agent."""
    query_id: str
    agent: str
    question: str
    contexts: list[str]          # retrieved chunks (RAG); empty if not RAG
    ground_truth: str | None = None
    # routing / calibration metadata (from calib_clean.yaml)
    expected_agents: list[str] = field(default_factory=list)
    difficulty: str = ""         # "simple" | "medium" | "complex"
    risk_level: str = ""         # "low" | "medium" | "high"
    category: str = ""
    high_risk: bool = False
    # gold retrieval ids (from calib_clean_with_gold.yaml); resolved to context
    # text via a CorpusStore when one is supplied.
    gold_law_ids: list[str] = field(default_factory=list)
    gold_case_ids: list[str] = field(default_factory=list)
    # ids of the documents the retriever ACTUALLY returned for this query (parallel
    # to `contexts`). Lets context_precision/recall be computed by set comparison
    # against the gold ids -- no LLM judge needed.
    retrieved_ids: list[str] = field(default_factory=list)


@dataclass
class QualityScores:
    quality: float                       # scalar aggregate in [0,1] -> MILP objective
    faithfulness: float = float("nan")
    answer_relevancy: float = float("nan")
    context_precision: float = float("nan")
    context_recall: float = float("nan")
    correctness: float = float("nan")
    # routing-specific (dispatcher only); None for specialists
    predicted_agents: list[str] | None = None
    routing_precision: float = float("nan")
    routing_recall: float = float("nan")


def aggregate(faithfulness=float("nan"), answer_relevancy=float("nan"),
              context_precision=float("nan"), context_recall=float("nan"),
              correctness=float("nan"),
              w_correctness: float = 0.5) -> float:
    """Combine GENERATION-only sub-scores into a single Q in [0,1].
    Retrieval metrics (context_precision/recall) are accepted in the signature for
    backward compatibility but are NOT used: they characterise the retriever, not
    the candidate, and are reported separately as retriever diagnostics. NaNs are
    skipped and weights renormalised."""
    grounding = [faithfulness, answer_relevancy]      # generation-only signals
    grounding = [g for g in grounding if not math.isnan(g)]
    g_mean = sum(grounding) / len(grounding) if grounding else float("nan")

    have_c = not math.isnan(correctness)
    have_g = not math.isnan(g_mean)
    if have_c and have_g:
        return w_correctness * correctness + (1 - w_correctness) * g_mean
    if have_c:
        return correctness
    if have_g:
        return g_mean
    return float("nan")


class QualityScorer:
    def score(self, sample: Sample, answer: str) -> QualityScores:
        raise NotImplementedError


class MockScorer(QualityScorer):
    """Deterministic quality from a per-config hint + answer length proxy."""

    def __init__(self, quality_hint: float = 0.5):
        self.quality_hint = quality_hint

    def score(self, sample: Sample, answer: str) -> QualityScores:
        base = self.quality_hint
        jitter = (len(answer) % 13) / 100.0
        f = min(1.0, base + 0.10 + jitter)
        r = min(1.0, base + 0.05)
        cp = min(1.0, base)
        cr = min(1.0, base - 0.02)
        corr = min(1.0, max(0.0, base - 0.05))
        q = aggregate(f, r, cp, cr, corr)
        return QualityScores(quality=q, faithfulness=f, answer_relevancy=r,
                             context_precision=cp, context_recall=cr, correctness=corr)


class RagasScorer(QualityScorer):
    """RAGAS-backed scorer. Requires `pip install ragas datasets` and a judge LLM.

    Pass a configured judge (e.g. an OpenAI-compatible model, or a strong LOCAL
    model via llama-cpp/Ollama for fully-offline scoring). `correctness_fn` is an
    optional callable(sample, answer)->float in [0,1] for the legal-correctness
    term that RAGAS alone does not capture.
    """

    def __init__(self, judge_llm=None, embeddings=None,
                 correctness_fn=None, w_correctness: float = 0.5):
        try:
            import ragas  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install ragas datasets  (and configure a judge LLM)") from e
        self.judge_llm = judge_llm
        self.embeddings = embeddings
        self.correctness_fn = correctness_fn
        self.w_correctness = w_correctness

    def score(self, sample: Sample, answer: str) -> QualityScores:  # pragma: no cover
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                    context_precision, context_recall)

        row = {
            "question": [sample.question],
            "answer": [answer],
            "contexts": [sample.contexts or [""]],
            "ground_truth": [sample.ground_truth or ""],
        }
        metrics = [faithfulness, answer_relevancy, context_precision]
        if sample.ground_truth:
            metrics.append(context_recall)
        res = evaluate(Dataset.from_dict(row), metrics=metrics,
                       llm=self.judge_llm, embeddings=self.embeddings)
        d = res.to_pandas().iloc[0].to_dict()
        corr = float("nan")
        if self.correctness_fn is not None:
            corr = float(self.correctness_fn(sample, answer))
        q = aggregate(
            faithfulness=float(d.get("faithfulness", float("nan"))),
            answer_relevancy=float(d.get("answer_relevancy", float("nan"))),
            context_precision=float(d.get("context_precision", float("nan"))),
            context_recall=float(d.get("context_recall", float("nan"))),
            correctness=corr, w_correctness=self.w_correctness,
        )
        return QualityScores(
            quality=q,
            faithfulness=float(d.get("faithfulness", float("nan"))),
            answer_relevancy=float(d.get("answer_relevancy", float("nan"))),
            context_precision=float(d.get("context_precision", float("nan"))),
            context_recall=float(d.get("context_recall", float("nan"))),
            correctness=corr,
        )


class JudgeScorer(QualityScorer):
    """Offline specialist scorer using a LOCAL judge model (an InferenceBackend,
    e.g. a strong llama.cpp model). No cloud, honoring the confidentiality premise.

    Computes, in a single judge call returning JSON:
      * correctness    -- answer vs ground_truth        (needs ground_truth; the
                          legally-important term RAGAS does NOT capture)
      * faithfulness   -- answer grounded in contexts    (only if contexts present)
      * context_precision / context_recall               (only if contexts present)
      * answer_relevancy -- answer addresses the question
    Missing inputs -> that metric is NaN; aggregate() skips NaNs. So this runs with
    JUST ground_truth (correctness-only) and lights up the grounding metrics once a
    corpus supplies the context text.

    This is a self-contained, dependency-light equivalent of the RAGAS metric family
    (no ragas/langchain needed). For the literal RAGAS library, use RagasScorer.
    """

    def __init__(self, judge_backend, w_correctness: float = 0.5, max_tokens: int = 400):
        self.judge = judge_backend
        self.w_correctness = w_correctness
        self.max_tokens = max_tokens

    def _prompt(self, sample: "Sample", answer: str) -> str:
        ctx = "\n".join(f"[CTX {i}] {c}" for i, c in enumerate(sample.contexts)) \
            if sample.contexts else "(no contexts provided)"
        gt = sample.ground_truth or "(no ground truth provided)"
        return (
            "You are a strict evaluator of an Italian family-law assistant. "
            "Score the ANSWER on a 0.0-1.0 scale for each metric below. "
            "Use null for a metric you cannot judge from the information given.\n"
            "Metrics:\n"
            "- correctness: does the ANSWER match the GROUND TRUTH (legally accurate)?\n"
            "- faithfulness: are the ANSWER's claims supported by the CONTEXTS?\n"
            "- context_precision: are the CONTEXTS relevant to the QUESTION?\n"
            "- context_recall: is the GROUND TRUTH covered by the CONTEXTS?\n"
            "- answer_relevancy: does the ANSWER address the QUESTION?\n\n"
            f"QUESTION:\n{sample.question}\n\n"
            f"CONTEXTS:\n{ctx}\n\n"
            f"GROUND TRUTH:\n{gt}\n\n"
            f"ANSWER:\n{answer}\n\n"
            "Reply with ONLY a JSON object, no prose, e.g.:\n"
            '{"correctness":0.0,"faithfulness":0.0,"context_precision":0.0,'
            '"context_recall":0.0,"answer_relevancy":0.0}'
        )

    def score(self, sample: "Sample", answer: str) -> QualityScores:
        try:
            res = self.judge.generate(self._prompt(sample, answer),
                                      max_tokens=self.max_tokens)
            d = _parse_score_json(res.text)
        except Exception as e:  # judge failed -> all NaN (derive will drop if needed)
            print(f"[judge] scoring failed for {sample.query_id}: {type(e).__name__}: {e}")
            d = {}

        def g(key):
            v = d.get(key)
            try:
                return float(v) if v is not None else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        corr = g("correctness")
        faith = g("faithfulness")
        cp = g("context_precision")
        cr = g("context_recall")
        ar = g("answer_relevancy")
        q = aggregate(faithfulness=faith, answer_relevancy=ar,
                      context_precision=cp, context_recall=cr,
                      correctness=corr, w_correctness=self.w_correctness)
        return QualityScores(quality=q, faithfulness=faith, answer_relevancy=ar,
                             context_precision=cp, context_recall=cr, correctness=corr)


def _parse_score_json(text: str) -> dict:
    """Pull the first {...} object out of the judge output and parse it. Robust to
    leading/trailing prose or code fences."""
    import json
    if not text:
        return {}
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return {}
    blob = text[s:e + 1]
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class MockJudgeBackend:
    """Deterministic judge for tests/dry runs: emits a fixed JSON of scores."""

    def __init__(self, scores: dict | None = None):
        self.scores = scores or {"correctness": 0.7, "faithfulness": 0.8,
                                 "context_precision": 0.75, "context_recall": 0.7,
                                 "answer_relevancy": 0.85}

    def generate(self, prompt: str, *, max_tokens: int = 400, context=None):
        import json
        from part1_allocation.inference.backend import GenResult
        txt = json.dumps(self.scores)
        return GenResult(text=txt, n_out_tokens=len(txt) // 4, ttft_s=0.01,
                         total_s=0.05, mean_logprob=-0.2)

    def close(self):
        pass


class SpecialistScorer(QualityScorer):
    """Judge-free specialist scorer (the default path).

    Uses two cheap, deterministic signals:

      * an EMBEDDER (e.g. bge-m3) for the three "semantic similarity" metrics:
          - correctness       = cos(answer, ground_truth_answer)
          - answer_relevancy  = cos(answer, question)
          - faithfulness      = max-pooled cos(answer, each_context), i.e. how
            well-anchored the answer is in the retrieved passages on average.

      * id-set comparison vs gold_*_ids for the two retrieval metrics:
          - context_precision = |retrieved n gold| / |retrieved|
          - context_recall    = |retrieved n gold| / |gold|

    NO LLM JUDGE NEEDED. faithfulness via embeddings is an APPROXIMATION (a judge
    would catch invented specifics a cosine can't); it's a deliberate trade for
    speed and determinism. Pass `judge_backend=` to override correctness/grounding
    with a judge -- the embedding values are kept as fallback for missing fields.
    """

    def __init__(self, judge_backend=None, embedder=None, w_correctness: float = 0.5,
                 include_correctness: bool = True, corr_embedder=None,
                 correctness_only: bool = False):
        """Judge-free specialist scorer.

        Parameters
        ----------
        include_correctness : bool, default True
            If True, the aggregated Q includes cos(answer, ground_truth_answer)
            with weight `w_correctness`.
            If False, Q is computed only from grounding signals (faithfulness +
            answer_relevancy), and the correctness term -- though still MEASURED
            and saved to the parquet as q_correctness -- is NOT folded into Q.
            Set False for specialists in domains where the gold ground truth is
            written for the FULL composed answer (covering multiple agents'
            contributions), so a single specialist's partial answer would be
            unfairly penalised by direct comparison. The synthesiser should
            keep this True (it produces the full user-visible answer).
        correctness_only : bool, default False
            If True, Q = correctness ALONE (faithfulness/relevancy excluded from
            the aggregate, though still measured and saved). Intended for the
            synthesiser: it is fed the real specialist answers and produces the
            final user-visible answer, so cos(answer, gold) already reflects every
            upstream error; mixing in grounding would only dilute that signal.
            Takes precedence over include_correctness when set.
        corr_embedder : Embedder or None, default None
            OPTIONAL separate embedder used ONLY for the correctness cosine
            (Tier-3 Finding 6). When None, correctness reuses `embedder` -- the
            same model as retrieval -- which makes faithfulness and correctness
            share the retriever's geometry (a model that parrots retrieved text
            scores high on both without being legally correct). Passing a
            DIFFERENT embedder here breaks that circularity. `embedder` is still
            used for answer_relevancy and faithfulness.
        """
        self.judge = JudgeScorer(judge_backend) if judge_backend is not None else None
        self.embedder = embedder
        self.corr_embedder = corr_embedder if corr_embedder is not None else embedder
        self.w_correctness = w_correctness
        self.include_correctness = include_correctness
        self.correctness_only = correctness_only

    def score(self, sample: "Sample", answer: str) -> QualityScores:
        # optional judge override (off by default)
        js = self.judge.score(sample, answer) if self.judge is not None else None

        # --- embedding-based: correctness, answer_relevancy, faithfulness ----
        emb_rel = float("nan")
        emb_corr = float("nan")
        emb_faith = float("nan")
        if self.embedder is not None and answer:
            from .embeddings import cosine
            qa = self.embedder.encode([answer, sample.question])
            emb_rel = max(0.0, cosine(qa[0], qa[1]))
            if sample.ground_truth:
                ga = self.corr_embedder.encode([answer, sample.ground_truth])
                emb_corr = max(0.0, cosine(ga[0], ga[1]))
            if sample.contexts:
                vecs = self.embedder.encode([answer] + list(sample.contexts))
                ans_v, ctx_v = vecs[0], vecs[1:]
                # max similarity to any context: a claim is "supported" if at least
                # one retrieved passage backs it. Mean would unfairly penalise multi-
                # context retrieval where most chunks are background.
                emb_faith = max(0.0, max(float(cosine(ans_v, c)) for c in ctx_v))

        # --- id-set based: context_precision / context_recall ----------------
        cp = float("nan")
        cr = float("nan")
        gold_ids = set(sample.gold_law_ids) | set(sample.gold_case_ids)
        ret_ids = set(sample.retrieved_ids)
        if ret_ids and gold_ids:
            inter = ret_ids & gold_ids
            cp = len(inter) / len(ret_ids)
            cr = len(inter) / len(gold_ids)
        elif gold_ids and not ret_ids:
            # No retriever ran (oracle contexts = gold passages themselves)
            # -> precision/recall are trivially 1 by construction.
            cp = 1.0
            cr = 1.0

        # judge values override embedding ones when present and finite
        correctness = js.correctness if (js and not _isnan(js.correctness)) else emb_corr
        answer_relevancy = (js.answer_relevancy if (js and not _isnan(js.answer_relevancy))
                            else emb_rel)
        faithfulness = js.faithfulness if (js and not _isnan(js.faithfulness)) else emb_faith
        cp = js.context_precision if (js and not _isnan(js.context_precision)) else cp
        cr = js.context_recall if (js and not _isnan(js.context_recall)) else cr

        # Q is the aggregated objective passed to the MILP.
        # If include_correctness=False, exclude it from the aggregate but STILL
        # save its measured value in the QualityScores -> parquet, so post-hoc
        # ablations can re-include it without remeasurement.
        if self.correctness_only:
            # synth: Q = correctness alone (grounding still measured & saved).
            q = correctness
        elif self.include_correctness:
            q = aggregate(faithfulness=faithfulness, answer_relevancy=answer_relevancy,
                          context_precision=cp, context_recall=cr,
                          correctness=correctness, w_correctness=self.w_correctness)
        else:
            q = aggregate(faithfulness=faithfulness, answer_relevancy=answer_relevancy,
                          context_precision=cp, context_recall=cr,
                          correctness=float("nan"),
                          w_correctness=self.w_correctness)
        return QualityScores(quality=q, faithfulness=faithfulness,
                             answer_relevancy=answer_relevancy, context_precision=cp,
                             context_recall=cr, correctness=correctness)


def _isnan(x) -> bool:
    import math
    return isinstance(x, float) and math.isnan(x)


# --------------------------------------------------------------------------- #
# NOT RAGAS. expected_agents is multi-label, so we use multi-label P/R/F1.
# --------------------------------------------------------------------------- #
def multilabel_prf(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Set-based precision / recall / F1 for multi-label routing."""
    if not expected and not predicted:
        return 1.0, 1.0, 1.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def parse_predicted_agents(text: str, known_agents: list[str]) -> set[str]:
    """Extract agent ids the dispatcher named in its output.
    Convention: prompt the dispatcher to emit the agent ids verbatim."""
    return {a for a in known_agents if a in (text or "")}


class RoutingScorer(QualityScorer):
    """Real dispatcher scorer: parse predicted agent ids from the model output,
    compare to sample.expected_agents. quality = F1 -> the MILP objective for the
    dispatcher agent."""

    def __init__(self, known_agents: list[str]):
        self.known_agents = list(known_agents)

    def score(self, sample: Sample, answer: str) -> QualityScores:
        predicted = parse_predicted_agents(answer, self.known_agents)
        expected = set(sample.expected_agents)
        p, r, f1 = multilabel_prf(predicted, expected)
        return QualityScores(quality=f1, predicted_agents=sorted(predicted),
                             routing_precision=p, routing_recall=r)


class MockRoutingScorer(QualityScorer):
    """Deterministic routing simulation for dry runs. Better dispatcher configs
    (higher quality_hint) recover the expected set more faithfully."""

    def __init__(self, known_agents: list[str], quality_hint: float = 0.5,
                 config_tag: str = "mock"):
        self.known_agents = list(known_agents)
        self.quality_hint = quality_hint
        self.config_tag = config_tag

    def _rng(self, sample: Sample):
        import random
        seed = int(hashlib.sha256(
            (self.config_tag + "|" + sample.query_id).encode()).hexdigest()[:8], 16)
        return random.Random(seed)

    def score(self, sample: Sample, answer: str) -> QualityScores:
        rng = self._rng(sample)
        expected = list(sample.expected_agents)
        predicted: set[str] = set()
        # keep each true agent with prob ~ quality_hint (harder on complex queries)
        keep_p = self.quality_hint - (0.15 if sample.difficulty == "complex" else 0.0)
        for a in expected:
            if rng.random() < max(0.1, keep_p):
                predicted.add(a)
        # occasionally hallucinate a wrong agent (less often for better configs)
        if rng.random() > self.quality_hint:
            others = [a for a in self.known_agents if a not in expected]
            if others:
                predicted.add(rng.choice(others))
        if not predicted and expected:        # never emit empty if there was a target
            predicted.add(expected[0])
        p, r, f1 = multilabel_prf(predicted, set(expected))
        return QualityScores(quality=f1, predicted_agents=sorted(predicted),
                             routing_precision=p, routing_recall=r)
