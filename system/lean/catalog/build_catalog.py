"""Build `lean/catalog/catalog.json` by joining:

- `lean/catalog/catalog.yaml`         — the Qwen2.5 universe (3×3×3 = 27 triples)
- `mamap_repo/shared/data/perf_table.parquet`  — measured ttft / throughput / energy
- a small Qwen2.5 architecture table — n_layers, n_kv_heads, head_dim (for κ_k)

The output JSON is the single artifact every downstream module reads, with a
sibling `.sha256` for reproducibility. A row is silently dropped if no
matching row exists in the perf table, and the dropped triples are reported
to stdout so the user can see them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Allow running as `python -m catalog.build_catalog` from `mamap_repo/lean/`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.types import Catalog, Config, ModelQuantGroup

_LEAN_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _LEAN_ROOT.parent
def _find_shared(*names):
    """Locate a shared data file across known layouts (shared/data or shared/pareto)."""
    for sub in ("data", "pareto"):
        for nm in names:
            p = _REPO_ROOT / "shared" / sub / nm
            if p.exists():
                return p
    return _REPO_ROOT / "shared" / "data" / names[0]   # default (may not exist yet)
_PERF_TABLE = _find_shared("perf_table.parquet")
_CATALOG_YAML = _LEAN_ROOT / "catalog" / "catalog.yaml"
_CATALOG_JSON = _LEAN_ROOT / "catalog" / "catalog.json"


# Qwen2.5 architecture facts (sourced from each model's `config.json` on HF).
# Used to compute κ_k = 2·L·n_kv·d·c·b_kv per the closed form.
@dataclass(frozen=True)
class _Arch:
    params: int
    n_layers: int
    n_kv_heads: int
    head_dim: int
    c_max: int


_QWEN_ARCH: dict[str, _Arch] = {
    "Qwen/Qwen2.5-0.5B-Instruct": _Arch(
        params=494_032_768, n_layers=24, n_kv_heads=2, head_dim=64, c_max=32_768
    ),
    "Qwen/Qwen2.5-1.5B-Instruct": _Arch(
        params=1_543_714_304, n_layers=28, n_kv_heads=2, head_dim=128, c_max=32_768
    ),
    "Qwen/Qwen2.5-3B-Instruct": _Arch(
        params=3_085_938_688, n_layers=36, n_kv_heads=2, head_dim=128, c_max=32_768
    ),
}


def _load_spec(yaml_path: Path) -> tuple[list[dict[str, str]], list[str], list[int]]:
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return (
        list(spec["models"]),
        list(spec["quants"]),
        [int(c) for c in spec["contexts"]],
    )


def _load_perf(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # The shared table carries one `hardware` column (all rows from the same
    # RTX 4070 Laptop). We index by `config_id` and ignore hardware here.
    required = {"config_id", "peak_mem_gb", "ttft_s", "throughput_tok_s", "energy_j_per_tok"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"perf_table missing columns: {missing}")
    return df.set_index("config_id")


def _config_id(perf_prefix: str, quant: str, context_length: int) -> str:
    return f"{perf_prefix}__{quant}__c{context_length}"


def _weight_gb_from_perf(peak_mem_gb: float, kv_gb: float) -> float:
    """Recover w_g from peak_mem_gb − κ_k.

    The perf table reports total peak VRAM (weights + KV + framework overhead).
    Subtracting the closed-form KV yields a w_g proxy that is **per-config** —
    we then take the median across contexts within a group to fix one w_g per
    `(model, quant)` so the load-once accounting is well-defined.
    """
    return max(0.01, peak_mem_gb - kv_gb)


def build() -> Catalog:
    models, quants, contexts = _load_spec(_CATALOG_YAML)
    if not _PERF_TABLE.exists():
        raise FileNotFoundError(
            f"Shared perf table missing: {_PERF_TABLE}.\n"
            f"  Did you clone the colleague's `shared/` directory?"
        )
    perf = _load_perf(_PERF_TABLE)

    # First pass: per-config weight estimates and KV.
    group_w_samples: dict[tuple[str, str], list[float]] = {}
    rows: list[dict[str, Any]] = []
    dropped: list[str] = []

    for entry in models:
        hf_id: str = entry["hf_id"]
        perf_prefix: str = entry["perf_prefix"]
        if hf_id not in _QWEN_ARCH:
            dropped.append(f"{hf_id}: no architecture record")
            continue
        arch = _QWEN_ARCH[hf_id]
        for quant in quants:
            group_w_samples.setdefault((hf_id, quant, perf_prefix), [])
            for ctx in contexts:
                if ctx > arch.c_max:
                    dropped.append(f"{hf_id} {quant} ctx={ctx}: above c_max")
                    continue
                cid = _config_id(perf_prefix, quant, ctx)
                if cid not in perf.index:
                    dropped.append(f"{cid}: not in perf_table")
                    continue
                row = perf.loc[cid]
                kv_bytes = 2 * arch.n_layers * arch.n_kv_heads * arch.head_dim * ctx * 2.0
                kv_gb = kv_bytes / (1024**3)
                w_gb = _weight_gb_from_perf(float(row["peak_mem_gb"]), kv_gb)
                group_w_samples[(hf_id, quant, perf_prefix)].append(w_gb)
                rows.append(
                    {
                        "hf_id": hf_id,
                        "quant": quant,
                        "context_length": ctx,
                        "ttft_s": float(row["ttft_s"]),
                        "throughput_tps": float(row["throughput_tok_s"]),
                        "energy_j_per_tok": float(row["energy_j_per_tok"]),
                        "arch": arch,
                    }
                )

    # Second pass: pick one w_g per group (median over contexts) and assemble groups.
    groups: dict[tuple[str, str], ModelQuantGroup] = {}
    for (hf_id, quant, perf_prefix), samples in group_w_samples.items():
        if not samples:
            continue
        w_g = float(pd.Series(samples).median())
        arch = _QWEN_ARCH[hf_id]
        groups[(hf_id, quant)] = ModelQuantGroup(
            model_id=hf_id,
            quant=quant,
            params=arch.params,
            weight_gb=w_g,
            c_max=arch.c_max,
            n_layers=arch.n_layers,
            n_kv_heads=arch.n_kv_heads,
            head_dim=arch.head_dim,
            perf_prefix=perf_prefix,
        )

    # Third pass: assemble Config objects.
    configs: list[Config] = []
    for r in rows:
        g = groups.get((r["hf_id"], r["quant"]))
        if g is None:
            continue
        configs.append(
            Config(
                group=g,
                context_length=int(r["context_length"]),
                ttft_s=r["ttft_s"],
                throughput_tps=r["throughput_tps"],
                energy_j_per_tok=r["energy_j_per_tok"],
            )
        )
    configs.sort(key=lambda c: (c.model_id, c.quant, c.context_length))

    if dropped:
        print(f"[build_catalog] dropped {len(dropped)} row(s):", file=sys.stderr)
        for d in dropped:
            print(f"  - {d}", file=sys.stderr)

    return Catalog(configs=tuple(configs))


def write_json(catalog: Catalog, out_path: Path) -> str:
    payload = {
        "meta": {"source": "shared/data/perf_table.parquet", "spec": "catalog.yaml"},
        "configs": [
            {
                "hf_id": c.group.model_id,
                "quant": c.group.quant,
                "context_length": c.context_length,
                "params": c.group.params,
                "weight_gb": c.group.weight_gb,
                "c_max": c.group.c_max,
                "n_layers": c.group.n_layers,
                "n_kv_heads": c.group.n_kv_heads,
                "head_dim": c.group.head_dim,
                "kv_dtype_bytes": c.group.kv_dtype_bytes,
                "perf_prefix": c.group.perf_prefix,
                "ttft_s": c.ttft_s,
                "throughput_tps": c.throughput_tps,
                "energy_j_per_tok": c.energy_j_per_tok,
            }
            for c in catalog.configs
        ],
    }
    out = json.dumps(payload, indent=2, sort_keys=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    digest = hashlib.sha256(out.encode("utf-8")).hexdigest()
    out_path.with_suffix(out_path.suffix + ".sha256").write_text(digest, encoding="utf-8")
    return digest


def read_json(path: Path, verify_hash: bool = True) -> Catalog:
    payload = path.read_text(encoding="utf-8")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if verify_hash and sidecar.exists():
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if expected != actual:
            raise ValueError(
                f"Catalog hash mismatch for {path.name}: "
                f"expected {expected[:12]}…, got {actual[:12]}…"
            )
    data = json.loads(payload)
    groups: dict[tuple[str, str], ModelQuantGroup] = {}
    configs: list[Config] = []
    for e in data["configs"]:
        key = (e["hf_id"], e["quant"])
        g = groups.get(key)
        if g is None:
            g = ModelQuantGroup(
                model_id=e["hf_id"],
                quant=e["quant"],
                params=int(e["params"]),
                weight_gb=float(e["weight_gb"]),
                c_max=int(e["c_max"]),
                n_layers=int(e["n_layers"]),
                n_kv_heads=int(e["n_kv_heads"]),
                head_dim=int(e["head_dim"]),
                kv_dtype_bytes=float(e.get("kv_dtype_bytes", 2.0)),
                perf_prefix=e.get("perf_prefix"),
            )
            groups[key] = g
        configs.append(
            Config(
                group=g,
                context_length=int(e["context_length"]),
                ttft_s=float(e["ttft_s"]),
                throughput_tps=float(e["throughput_tps"]),
                energy_j_per_tok=float(e["energy_j_per_tok"]),
            )
        )
    return Catalog(configs=tuple(configs))


def main() -> None:
    catalog = build()
    digest = write_json(catalog, _CATALOG_JSON)
    print(
        f"[build_catalog] wrote {len(catalog)} configs from "
        f"{len(catalog.groups)} groups → {_CATALOG_JSON.name} "
        f"(sha256={digest[:12]}…)"
    )


if __name__ == "__main__":
    main()
