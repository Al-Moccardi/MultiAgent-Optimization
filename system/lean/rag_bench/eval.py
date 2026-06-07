"""Harness skeleton: iterate (role, config, query) → score → quality.parquet.

The actual eval is **out of scope** for the lean MVP — this module ships:

1. A `Generator` Protocol + `MockGenerator` for CI.
2. `score_one(generation, query)` wrapping the colleague's RAGAS scorer
   (`shared/faiss_code/scorer.py`) when available, with a deterministic
   analytical fallback otherwise.
3. `run(...)` that orchestrates the (role, config, query) loop and writes
   `quality.parquet` in the shape `src.quality.Quality.from_parquet` expects.

Real backends (vLLM HTTP, llama.cpp server, OpenAI-compatible) plug in by
implementing the `Generator` Protocol.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bench.build_subset import from_parquet
from rag_bench.types import Generation, Generator, Query, Subset

# ---------------------------------------------------------------------------
# Mock generator (CI-safe; never produces a real LLM answer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockGenerator:
    """Deterministic generator: echoes the gold context shortened to N words.

    Useful only for harness smoke tests — the scorer will give a passable
    cosine to the gold answer since the gold context overlaps lexically.
    """

    output_words: int = 32

    def generate(self, query: Query, role: str, config_id: str) -> Generation:
        words = query.gold_context.split()[: self.output_words]
        return Generation(
            query_id=query.query_id,
            domain=query.domain,
            role=role,
            config_id=config_id,
            answer=" ".join(words),
            wall_s=0.0,
            n_input_tokens=len(query.query.split()),
            n_output_tokens=len(words),
        )


# ---------------------------------------------------------------------------
# Scoring — RAGAS-aggregate via colleague's scorer, with an analytical fallback.
# ---------------------------------------------------------------------------


def _shared_scorer():
    """Try to import `shared/faiss_code/scorer.py:aggregate`. Returns None if unavailable."""
    repo_root = Path(__file__).resolve().parents[2]
    shared_path = repo_root / "shared" / "faiss_code"
    if not (shared_path / "scorer.py").exists():
        return None
    sys.path.insert(0, str(shared_path))
    try:
        from scorer import aggregate  # type: ignore[import-not-found]
    except Exception:
        return None
    return aggregate


def _fallback_score(generation: Generation, query: Query) -> float:
    """Deterministic, embedding-free fallback: token-set Jaccard with gold_answer."""
    a = set(generation.answer.lower().split())
    g = set(query.gold_answer.lower().split())
    if not a or not g:
        return 0.0
    return len(a & g) / len(a | g)


def score_one(generation: Generation, query: Query) -> float:
    """Aggregate score in [0, 1] for one generation; uses colleague's scorer if importable."""
    aggregate = _shared_scorer()
    if aggregate is None:
        return _fallback_score(generation, query)
    # The colleague's aggregate(faithfulness, answer_relevancy, context_precision,
    # context_recall, correctness, w_correctness=0.5) expects cosine-similarity
    # numbers in [0, 1]; we approximate them lexically here so the scorer is
    # exercised without bringing in sentence-transformers.
    f = _fallback_score(generation, query)
    return float(
        aggregate(
            faithfulness=f,
            answer_relevancy=f,
            context_precision=1.0,
            context_recall=1.0,
            correctness=f,
        )
    )


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------


def run(
    subset: Subset,
    config_ids: Iterable[str],
    generator: Generator,
    roles: tuple[str, str, str] = ("d", "s", "y"),
) -> pd.DataFrame:
    """For every (role, config, query): generate, score, accumulate.

    Returns a DataFrame with columns (role, config_id, query_id, domain, score)
    suitable for `df.to_parquet(out)` and `Quality.from_parquet(out)`.
    """
    rows: list[dict] = []
    cfgs = list(config_ids)
    for role in roles:
        for cid in cfgs:
            for q in subset.queries:
                gen = generator.generate(q, role, cid)
                s = score_one(gen, q)
                rows.append(
                    {
                        "role": role,
                        "config_id": cid,
                        "query_id": q.query_id,
                        "domain": q.domain,
                        "score": s,
                    }
                )
    return pd.DataFrame(rows)


def average_per_role_config_domain(scores: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-query scores to one row per (role, config, domain).

    This is the shape `Quality.from_parquet` consumes. For the dispatcher and
    synthesizer roles the `domain` column is dropped via aggregation since
    those roles aren't domain-conditional.
    """
    by_role: list[pd.DataFrame] = []
    for role in ("d", "y"):
        sub = scores[scores["role"] == role]
        if not sub.empty:
            agg = sub.groupby("config_id", as_index=False)["score"].mean()
            agg["role"] = role
            agg["domain"] = ""
            by_role.append(agg[["role", "config_id", "domain", "score"]])
    sub_s = scores[scores["role"] == "s"]
    if not sub_s.empty:
        agg = sub_s.groupby(["config_id", "domain"], as_index=False)["score"].mean()
        agg["role"] = "s"
        by_role.append(agg[["role", "config_id", "domain", "score"]])
    return pd.concat(by_role, ignore_index=True) if by_role else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--config-ids",
        type=str,
        required=True,
        help="Comma-separated list of config_ids to score",
    )
    args = p.parse_args(argv)

    subset = from_parquet(args.subset)
    config_ids = [c.strip() for c in args.config_ids.split(",") if c.strip()]
    df = run(subset, config_ids, generator=MockGenerator())
    out = average_per_role_config_domain(df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(
        f"[rag_bench] wrote {len(out)} (role, config, domain) rows → {args.out.name}"
    )


if __name__ == "__main__":
    main()
