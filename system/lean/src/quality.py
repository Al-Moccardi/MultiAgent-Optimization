"""Quality lookup — the per-role scores the MILP maximises.

In production (L5) this is loaded from `lean/catalog/quality.parquet`, written
by `multihop_rag/eval.py` (RAGAS aggregate via `shared/faiss_code/scorer.py`).
Here we also ship a synthetic generator so the MILP can be tested end-to-end
before the real harness lands.

Shape:

    F_d : config_id → float in [0, 1]              dispatcher F1
    Q_s : (config_id, domain) → float in [0, 1]    specialist per-domain
    Q_y : config_id → float in [0, 1]              synthesizer composite

The MILP objective is

    Q = F_d(c_d)
        + (1/|D|) · Σ_{δ∈D} Q_s(c_s, δ)
        + Q_y(c_y)

where `D` is the per-instance specialist-domain set (`Instance.domains`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.types import Catalog


@dataclass(frozen=True)
class Quality:
    F_d: dict[str, float] = field(default_factory=dict)
    Q_s: dict[tuple[str, str], float] = field(default_factory=dict)
    Q_y: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_parquet(cls, path: Path) -> Quality:
        """Load from `quality.parquet` with columns (role, config_id, domain, score)."""
        df = pd.read_parquet(path)
        required = {"role", "config_id", "score"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"quality.parquet missing columns: {missing}")
        F_d: dict[str, float] = {}
        Q_s: dict[tuple[str, str], float] = {}
        Q_y: dict[str, float] = {}
        for row in df.itertuples(index=False):
            role = row.role
            cid = row.config_id
            score = float(row.score)
            if role == "d":
                F_d[cid] = score
            elif role == "y":
                Q_y[cid] = score
            elif role == "s":
                domain = getattr(row, "domain", None) or ""
                Q_s[(cid, domain)] = score
        return cls(F_d=F_d, Q_s=Q_s, Q_y=Q_y)


# ---------------------------------------------------------------------------
# Synthetic quality generator (analytical, for tests + smoke runs)
# ---------------------------------------------------------------------------

# Per-role base scaling: bigger model → higher dispatcher F1 and synth Q (these
# are sortable signals); specialist Q is non-monotone in size (matches the
# colleague's finding) and is highest near the 1.5B sweet spot.
_DISP_F1_BY_PARAMS_B = lambda p: float(np.clip(0.30 + 0.10 * np.log10(p), 0.30, 0.80))  # noqa: E731
_SYNTH_BY_PARAMS_B = lambda p: float(np.clip(0.45 + 0.08 * np.log10(p), 0.45, 0.75))  # noqa: E731


def _spec_quality_by_params_b(params_b: float, domain: str) -> float:
    """Non-monotone in size, peak near ~1.5B, with per-domain noise."""
    peak = 1.5
    bell = float(np.exp(-((params_b - peak) ** 2) / (2 * 1.0**2)))  # σ ≈ 1.0
    base = 0.50 + 0.20 * bell
    # Deterministic per-domain offset so swapping domains shuffles scores
    # without injecting RNG noise.
    domain_offset = (sum(ord(c) for c in domain) % 7 - 3) * 0.02
    return float(np.clip(base + domain_offset, 0.0, 1.0))


# Quantisation penalty: lower bits → slight quality drop. Q3 < Q5 < Q8 < F16.
_QUANT_PENALTY = {
    "Q3_K_M": 0.10,
    "Q5_K_M": 0.03,
    "Q8_0": 0.00,
    "F16": 0.00,
}


def synthetic_quality(catalog: Catalog, domains: tuple[str, ...] = ()) -> Quality:
    """Plausible synthetic scores for testing — never use these in the paper.

    Encodes the qualitative findings the colleague observed empirically:
    bigger models help dispatcher / synth, specialist quality peaks at ~1.5B.
    """
    if not domains:
        domains = ("default",)
    F_d: dict[str, float] = {}
    Q_s: dict[tuple[str, str], float] = {}
    Q_y: dict[str, float] = {}
    for cfg in catalog.configs:
        params_b = cfg.group.params / 1e9
        penalty = _QUANT_PENALTY.get(cfg.quant, 0.05)
        cid = cfg.config_id
        F_d[cid] = max(0.0, _DISP_F1_BY_PARAMS_B(params_b) - penalty)
        Q_y[cid] = max(0.0, _SYNTH_BY_PARAMS_B(params_b) - penalty)
        for d in domains:
            Q_s[(cid, d)] = max(0.0, _spec_quality_by_params_b(params_b, d) - penalty)
    return Quality(F_d=F_d, Q_s=Q_s, Q_y=Q_y)
