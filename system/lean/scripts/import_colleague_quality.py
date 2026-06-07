"""Subset the colleague's `quality_table.parquet` → lean `catalog/quality.parquet`.

**Path A** (whitepaper §7): re-use the colleague's already-measured per-(agent,
query, config) quality on the Italian family-law dataset, restricted to the
Qwen2.5 configs the lean catalog uses. Produces a `quality.parquet` in the
shape `src.quality.Quality.from_parquet` consumes, plus a sibling
`quality.meta.json` declaring provenance so the run is reproducible and the
data lineage is auditable.

Agent → role/domain mapping (colleague's 11 agents → lean 3 roles):

    A_dispatcher                                    → role d (single F1 number per config)
    A_synth                                         → role y (single RAGAS number per config)
    A_<rest>                                        → role s with domain=<rest>
                                                       (9 specialist domains from the
                                                       Italian family-law pipeline)

For roles `d` and `y` we take the mean `quality` across all queries the agent
was scored on. For role `s` we take the per-(config, specialist) mean across
the queries that specialist was *activated* on — which is the colleague's
ground-truth notion of "this specialist's quality on its domain" because the
specialist is only run on its in-domain queries.

Usage:
    python -m scripts.import_colleague_quality
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quality import Quality  # noqa: E402

_LEAN_ROOT = Path(__file__).resolve().parents[1]
def _find_quality():
    cands = [_LEAN_ROOT.parent / "shared" / "data" / "quality_table.parquet",
             _LEAN_ROOT.parent / "shared" / "pareto" / "quality_table.parquet",
             _LEAN_ROOT.parent / "part4_dynamic_path" / "data" / "quality_table.parquet",
             _LEAN_ROOT.parent / "part2_quality_aware" / "data" / "quality_table.parquet"]
    for p in cands:
        if p.exists():
            return p
    return cands[0]
_COLLEAGUE_PARQUET = _find_quality()
_OUT_PARQUET = _LEAN_ROOT / "catalog" / "quality.parquet"
_OUT_META = _LEAN_ROOT / "catalog" / "quality.meta.json"


def _agent_to_role_and_domain(agent: str) -> tuple[str, str]:
    """Colleague's agent name → lean (role, domain) pair.

    The dispatcher and synth produce a single number per config; specialists
    each represent one domain.
    """
    if agent == "A_dispatcher":
        return "d", ""
    if agent == "A_synth":
        return "y", ""
    if not agent.startswith("A_"):
        raise ValueError(f"unrecognised agent label '{agent}' (expected 'A_*')")
    return "s", agent[len("A_") :]


def subset(
    qt: pd.DataFrame,
    family_prefix: str = "qwen2.5-",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Filter to the lean family, aggregate to (role, config_id, domain, score).

    Returns the aggregated frame plus a per-key diagnostics dict (n_queries
    behind every aggregate) so the meta.json captures sample-size signal.
    """
    df = qt[qt["config_id"].str.startswith(family_prefix)].copy()
    if df.empty:
        raise ValueError(
            f"No rows in colleague's quality_table with config_id prefix "
            f"'{family_prefix}'. Did the colleague's catalog change?"
        )
    df = df.dropna(subset=["quality"])
    role_domain = df["agent"].map(_agent_to_role_and_domain)
    df["role"] = role_domain.map(lambda rd: rd[0])
    df["domain"] = role_domain.map(lambda rd: rd[1])

    agg = (
        df.groupby(["role", "config_id", "domain"], as_index=False)
        .agg(score=("quality", "mean"), n_queries=("query_id", "nunique"))
    )
    diagnostics: dict[str, dict] = {}
    for row in agg.itertuples(index=False):
        diagnostics[f"{row.role}|{row.config_id}|{row.domain}"] = {
            "n_queries": int(row.n_queries),
            "mean": float(row.score),
        }
    out = agg[["role", "config_id", "domain", "score"]].sort_values(
        ["role", "config_id", "domain"]
    ).reset_index(drop=True)
    return out, diagnostics


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_with_meta(
    df: pd.DataFrame,
    out_parquet: Path,
    out_meta: Path,
    source: Path,
    diagnostics: dict[str, dict],
    family_prefix: str,
) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    meta = {
        "generator": "scripts/import_colleague_quality.py",
        "generated_at": dt.datetime.now(tz=dt.UTC).isoformat(),
        "quality_source": "colleague:shared/data/quality_table.parquet",
        "source_path": str(source.relative_to(_LEAN_ROOT.parent.parent))
        if source.is_absolute()
        else str(source),
        "source_sha256": _file_sha256(source),
        "family_prefix": family_prefix,
        "n_rows_out": int(len(df)),
        "agent_to_role": {
            "A_dispatcher": "d",
            "A_synth": "y",
            "A_<specialist_*>": "s (domain = name after 'A_')",
        },
        "notes": (
            "Per role d / y: mean(quality) across queries the agent was scored on. "
            "Per role s: mean(quality) per (config, specialist) restricted to the "
            "queries the specialist was activated on — the colleague's ground-truth "
            "for 'this specialist's quality on its domain'."
        ),
        "n_queries_diagnostic": diagnostics,
    }
    out_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=_COLLEAGUE_PARQUET,
        help="colleague's quality_table.parquet",
    )
    p.add_argument("--out", type=Path, default=_OUT_PARQUET)
    p.add_argument("--meta", type=Path, default=_OUT_META)
    p.add_argument("--family-prefix", type=str, default="qwen2.5-")
    args = p.parse_args(argv)

    if not args.source.exists():
        raise FileNotFoundError(
            f"Colleague's quality_table is missing at {args.source}. "
            f"Check `mamap_repo/shared/data/`."
        )
    qt = pd.read_parquet(args.source)
    df, diagnostics = subset(qt, family_prefix=args.family_prefix)
    write_with_meta(
        df,
        args.out,
        args.meta,
        source=args.source,
        diagnostics=diagnostics,
        family_prefix=args.family_prefix,
    )

    # Sanity round-trip: re-read with Quality.from_parquet so we know the MILP
    # will be happy with the output.
    q = Quality.from_parquet(args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "meta": str(args.meta),
                "n_rows": len(df),
                "F_d_configs": len(q.F_d),
                "Q_y_configs": len(q.Q_y),
                "Q_s_pairs": len(q.Q_s),
                "domains": sorted({d for (_, d) in q.Q_s.keys()}),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
