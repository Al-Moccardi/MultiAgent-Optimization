"""
part4_dynamic_path/src/calibrate.py
===================================
OFFLINE calibration of the dynamic gate, from the calibration set ONLY
(gold routing labels). Two artefacts, both LLM-free:

  1. Relevance calibration: map a raw fused retrieval score s_d in [0,1] to a
     probability rho_hat_d = P(domain d is gold-relevant | s_d), via isotonic
     regression on (score, is_gold) pairs pooled over calibration queries and
     candidate domains. This makes the score a *probability*, not just a rank.

  2. Conformal coverage threshold (split conformal / risk control): pick the
     smallest inclusion threshold tau such that, with finite-sample validity,
     the policy's miss-rate of gold-relevant domains is <= alpha. Concretely we
     use the conformal-risk-control recipe: tau is the (1-alpha)-style empirical
     quantile of the *relevant-domain* scores on calibration, with the standard
     +1 finite-sample correction. Keeping a domain whenever rho_hat_d >= tau then
     controls the expected fraction of dropped gold domains at level alpha
     (marginal guarantee; see Angelopoulos & Bates 2023, Angelopoulos et al.
     2024 on conformal risk control).

Honest scope: with ~37 calibration root queries the guarantee is MARGINAL
(coverage holds on average over queries, not per query); we report achieved
coverage on the held-out test set so the gap is visible.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


# --------------------------------------------------------------------------- isotonic
class Isotonic:
    """Minimal PAV isotonic regression (monotone non-decreasing). No sklearn dep."""

    def __init__(self) -> None:
        self.x = np.array([0.0, 1.0])
        self.y = np.array([0.0, 1.0])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Isotonic":
        order = np.argsort(x)
        xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
        # pool-adjacent-violators
        w = np.ones_like(ys)
        val = ys.copy()
        i = 0
        blocks = [[val[j], w[j], xs[j], xs[j]] for j in range(len(ys))]
        merged = []
        for b in blocks:
            merged.append(b)
            while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
                v2, w2, lo2, hi2 = merged.pop()
                v1, w1, lo1, hi1 = merged.pop()
                nw = w1 + w2
                merged.append([(v1 * w1 + v2 * w2) / nw, nw, lo1, hi2])
        gx, gy = [], []
        for v, w_, lo, hi in merged:
            gx.append(lo); gy.append(v)
            gx.append(hi); gy.append(v)
        self.x = np.array(gx); self.y = np.clip(np.array(gy), 0.0, 1.0)
        return self

    def predict(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, float), self.x, self.y)


# --------------------------------------------------------------------------- calibration
@dataclass
class GateCalibration:
    isotonic: Isotonic
    tau: float            # conformal inclusion threshold on rho_hat
    alpha: float          # target miss-rate
    n_cal: int            # #calibration relevant pairs used

    def rho_hat(self, fused_score: float) -> float:
        return float(self.isotonic.predict([fused_score])[0])


def calibrate_gate(signal_rows: list[dict], alpha: float = 0.1) -> GateCalibration:
    """
    Parameters
    ----------
    signal_rows : list of {"score": float, "is_gold": 0/1} pooled over
        (calibration query, candidate domain) pairs. `score` is the fused
        retrieval score from signals.DomainRelevance.
    alpha : target miss-rate for gold-relevant domains.
    """
    s = np.array([r["score"] for r in signal_rows], float)
    g = np.array([r["is_gold"] for r in signal_rows], float)
    iso = Isotonic().fit(s, g)

    # conformal-risk-control threshold on calibrated probabilities of the
    # RELEVANT pairs: keep a domain if rho_hat >= tau, where tau is the
    # alpha-quantile of relevant rho_hats with the finite-sample (n+1) rule.
    rel = iso.predict(s[g == 1])
    n = len(rel)
    if n == 0:
        return GateCalibration(iso, tau=0.0, alpha=alpha, n_cal=0)
    rel_sorted = np.sort(rel)                      # ascending
    # index of the floor( alpha * (n+1) )-th smallest relevant score
    j = int(np.floor(alpha * (n + 1)))
    j = max(0, min(j, n - 1))
    tau = float(rel_sorted[j])
    return GateCalibration(iso, tau=tau, alpha=alpha, n_cal=n)


def build_calibration_signal_rows(relevance, calib_yaml: str | Path) -> list[dict]:
    """Run the relevance signal over every (calib query, every specialist) and
    label by gold membership -> the (score, is_gold) pool for calibration."""
    queries = yaml.safe_load(Path(calib_yaml).read_text(encoding="utf-8"))["queries"]
    rows: list[dict] = []
    for q in queries:
        gold = set(_parse(q.get("expected_agents")) or [])
        sc = relevance.scores(q["text"])  # all specialists
        for d, sig in sc.items():
            rows.append({"qid": q["id"], "domain": d,
                         "score": sig["fused"], "is_gold": int(d in gold)})
    return rows
