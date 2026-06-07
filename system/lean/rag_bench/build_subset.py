"""JSONL → stratified Subset → cached parquet.

Reads a domain-labelled JSONL (see rag_bench/README.md) and writes a parquet
the eval harness can consume. Stratified by `domain` so every domain is
represented even when `--max-per-domain` shrinks the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Make `rag_bench.*` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_bench.types import Query, Subset


def load_jsonl(path: Path) -> list[Query]:
    queries: list[Query] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(Query.model_validate_json(line))
    return queries


def stratify(
    queries: list[Query],
    max_per_domain: int | None = None,
    seed: int = 0,
) -> Subset:
    by_domain: dict[str, list[Query]] = defaultdict(list)
    for q in queries:
        by_domain[q.domain].append(q)

    import random

    rng = random.Random(seed)
    out: list[Query] = []
    for domain, qs in sorted(by_domain.items()):
        if max_per_domain is None or len(qs) <= max_per_domain:
            out.extend(qs)
        else:
            out.extend(rng.sample(qs, max_per_domain))
    return Subset(queries=tuple(out))


def to_parquet(subset: Subset, out_path: Path) -> None:
    rows = [q.model_dump() for q in subset.queries]
    pd.DataFrame(rows).to_parquet(out_path)


def from_parquet(path: Path) -> Subset:
    df = pd.read_parquet(path)
    queries = [Query(**row) for row in df.to_dict("records")]
    return Subset(queries=tuple(queries))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Domain-labelled JSONL")
    p.add_argument("--out", type=Path, required=True, help="Output parquet")
    p.add_argument(
        "--max-per-domain",
        type=int,
        default=None,
        help="Cap per-domain row count (stratified sample)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    queries = load_jsonl(args.input)
    subset = stratify(queries, max_per_domain=args.max_per_domain, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    to_parquet(subset, args.out)
    print(
        json.dumps(
            {
                "input_path": str(args.input),
                "out_path": str(args.out),
                "n_queries": len(subset.queries),
                "domains": list(subset.domains),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
