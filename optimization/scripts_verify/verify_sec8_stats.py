#!/usr/bin/env python3
"""Verify the §8 (measured-quality) claims of the paper from shared/data.

Paper claims verified here (expected values in parentheses):
  C1  Scored record counts: total 7959 = dispatcher 3900 + specialists 3439
      + synthesiser 620.
  C2  Pearson r between specialist quality and parameter count:
      0.54 at the configuration level, 0.31 at the per-query (record) level.
  C3  Best specialist model = Llama-3.2-1B (mean 0.74), beating Mistral-7B (0.72).
      Pooled Llama-3.2-1B: 282 records, 95% bootstrap CI [0.743, 0.764].
  C4  Quality-latency Pareto frontier over the 50 scored specialist
      configurations: exactly 4 non-dominated, none above 1.2 B params
      (llama3.2-1b Q5_K_M c4096 / c8192, smollm2-360m Q3_K_M c8192, Q8_0 c4096).
  C5  Per-domain winners: Llama-3.2-1B best in 7 of 9 domains; Mistral-7B best
      in negoziazione_assistita_e_mediazione_familiare and separazione_consensuale.
  C6  Synthesiser top-3 (scorecard): 0.768 / 0.729 / 0.717.
  C7  LODO (leave-one-domain-out): train-selected model = Llama-3.2-1B in 9/9
      folds; generalization gap vs per-fold oracle mean 0.001, max 0.008;
      beats Mistral-7B on the held-out domain in 7/9 folds (mean +0.018),
      losing only the two Mistral domains of C5.

Run:  python verify_sec8_stats.py        (from optimization/scripts_verify/)
"""
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "data"
rng = np.random.default_rng(0)

qt = pd.read_parquet(DATA / "quality_table.parquet")
qt = qt[qt.quality.notna()].copy()
sc = pd.read_csv(DATA / "quality_scorecard.csv")
perf = pd.read_parquet(DATA / "perf_table.parquet")
perf["model"] = perf.config_id.str.split("__").str[0]
perf["lat_gen"] = perf.ttft_s + 384 / perf.throughput_tok_s

zoo = yaml.safe_load(open(DATA / "catalog_zoo.yaml", encoding="utf-8"))
PAR = {m["key"]: float(m["params_b"]) for m in zoo["models"]}

SPEC = sorted(a for a in qt.agent.unique() if a not in ("A_dispatcher", "A_synth"))
qt["model"] = qt.config_id.str.split("__").str[0]
spec = qt[qt.agent.isin(SPEC)].copy()

# C1 ------------------------------------------------------------------ counts
n_d = int((qt.agent == "A_dispatcher").sum())
n_s = len(spec)
n_y = int((qt.agent == "A_synth").sum())
print(f"[C1] scored records: total={len(qt)}  dispatcher={n_d}  "
      f"specialists={n_s}  synth={n_y}   (expect 7959/3900/3439/620)")

# C2 ------------------------------------------------------------- correlations
# config level = scorecard per-config mean over specialist agents (the paper's
# aggregation); per-query level = raw specialist records.
spec["params"] = spec.model.map(PAR)
ssc0 = sc[~sc.agent.isin(["A_dispatcher", "A_synth"])]
cq0 = ssc0.groupby("config_id").quality.mean()
pp0 = cq0.index.str.split("__").str[0].map(PAR.get)
r_cfg = np.corrcoef(pp0, cq0)[0, 1]
r_rec = np.corrcoef(spec.params, spec.quality)[0, 1]
print(f"[C2] Pearson r quality~params: config-level={r_cfg:.2f} (expect 0.54)  "
      f"per-record={r_rec:.2f} (expect 0.31)")

# C3 --------------------------------------------------------------- best model
mm = (ssc0.assign(model=ssc0.config_id.str.split("__").str[0])
      .groupby("model").quality.mean().sort_values(ascending=False))
ll = spec[spec.model == "llama3.2-1b"].quality.to_numpy()
boot = [rng.choice(ll, len(ll), replace=True).mean() for _ in range(10000)]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"[C3] best={mm.index[0]} ({mm.iloc[0]:.2f})  mistral-7b={mm['mistral-7b']:.2f} "
      f"(expect llama3.2-1b 0.741 vs 0.725); pooled llama n={len(ll)}/{n_s}, "
      f"CI=[{lo:.3f},{hi:.3f}] (expect [0.743,0.764])")

# C4 ------------------------------------------------------------------- Pareto
ssc = sc[~sc.agent.isin(["A_dispatcher", "A_synth"])].copy()
cq = ssc.groupby("config_id").quality.mean()
lat = perf.set_index("config_id").lat_gen
pts = pd.DataFrame({"q": cq, "lat": lat}).dropna()
front = [c for c, row in pts.iterrows()
         if not ((pts.q >= row.q) & (pts.lat < row.lat) |
                 (pts.q > row.q) & (pts.lat <= row.lat)).any()]
pb = max(PAR[c.split("__")[0]] for c in front)
print(f"[C4] Pareto: {len(front)} of {len(pts)} non-dominated (expect 4 of 50), "
      f"max params {pb:.2f}B (expect <=1.2):")
for c in sorted(front):
    print(f"      {c}")

# C5 -------------------------------------------------------- per-domain winners
win = {}
for d in SPEC:
    win[d] = spec[spec.agent == d].groupby("model").quality.mean().idxmax()
n_ll = sum(1 for v in win.values() if v == "llama3.2-1b")
others = {d: m for d, m in win.items() if m != "llama3.2-1b"}
print(f"[C5] per-domain winners: llama3.2-1b in {n_ll}/9 (expect 7/9); others={others}")

# C6 ------------------------------------------------------------ synth ranking
syn = (sc[sc.agent == "A_synth"]
       .assign(model=lambda d: d.config_id.str.split("__").str[0])
       .groupby("model").quality.mean().sort_values(ascending=False).head(3))
print("[C6] synth model top-3 (expect mistral 0.768, phi3.5 0.729, gemma 0.717"
      " = paper's 0.77/0.73/0.72):")
for m, q in syn.items():
    print(f"      {m:16s} {q:.3f}")

# C7 --------------------------------------------------------------------- LODO
gaps, vs_mistral, losers, sel = [], [], [], []
for held in SPEC:
    tr = spec[spec.agent != held]
    te = spec[spec.agent == held]
    pick = tr.groupby("model").quality.mean().idxmax()
    sel.append(pick)
    teq = te.groupby("model").quality.mean()
    gaps.append(float(teq.max() - teq.get(pick, np.nan)))
    d = float(teq.get(pick, np.nan) - teq.get("mistral-7b", np.nan))
    vs_mistral.append(d)
    if d < 0:
        losers.append(held)
print(f"[C7] LODO: selected llama3.2-1b in {sel.count('llama3.2-1b')}/9 folds "
      f"(expect 9/9); gap mean={np.mean(gaps):.3f} max={np.max(gaps):.3f} "
      f"(expect 0.001/0.008); beats mistral in {sum(d>0 for d in vs_mistral)}/9 "
      f"(mean {np.mean(vs_mistral):+.3f}, expect 7/9 +0.018); losing folds={losers}")
print("DONE sec8")
