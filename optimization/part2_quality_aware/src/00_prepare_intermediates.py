"""
part2 / 00_prepare_intermediates.py
===================================
Builds the intermediate tables the part-2 analysis/plot scripts consume, from the
canonical inputs in shared/data. Run this FIRST in part 2.

Produces (in results/data/):
  - quality_cost_merged.parquet : per-(agent, config) quality joined with the
                                  384-token generation latency and params.
  - agg_frontier.parquet        : per-config specialist-mean quality + latency,
                                  with a pareto_lat flag (quality-vs-latency front).
  - dispatcher_f1.parquet       : per dispatcher MODEL, mean routing F1 + params.
  - proxy_vs_quality.csv        : parameter-proxy vs quality-optimal specialist at
                                  matched latency (k=3) -- the comparison table.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "shared" / "lib"))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)

import pandas as pd, numpy as np

PAR = _P.PARAMS_B
sc = pd.read_csv(str(_P.SCORECARD))
perf = pd.read_parquet(str(_P.PERF_TABLE))
perf["model"] = perf["config_id"].str.split("__").str[0]
perf["lat384"] = perf["ttft_s"] + _P.TOK_GENERATION / perf["throughput_tok_s"]
lat384 = {r.config_id: r.lat384 for r in perf.itertuples()}

# ---- quality_cost_merged: every (agent, config) score + latency + params ----
m = sc.copy()
m["model"] = m["config_id"].str.split("__").str[0]
m["params"] = m["model"].map(PAR)
m["lat384"] = m["config_id"].map(lat384)
m = m.dropna(subset=["lat384"])
m.to_parquet(str(_DATA / "quality_cost_merged.parquet"))
print(f"quality_cost_merged.parquet: {len(m)} rows")

# ---- agg_frontier: specialist-mean quality per config + pareto flag ----
SPEC = [a for a in sc["agent"].unique() if a not in ("A_dispatcher", "A_synth")]
spec = m[m["agent"].isin(SPEC)]
agg = (spec.groupby("config_id")
            .agg(quality=("quality", "mean"), params=("params", "first"),
                 lat384=("lat384", "first"))
            .reset_index())
# Pareto front (maximize quality, minimize latency)
agg = agg.sort_values("lat384").reset_index(drop=True)
best = -np.inf
flag = []
for _, r in agg.iterrows():
    if r["quality"] > best + 1e-9:
        flag.append(True); best = r["quality"]
    else:
        flag.append(False)
agg["pareto_lat"] = flag
agg.to_parquet(str(_DATA / "agg_frontier.parquet"))
print(f"agg_frontier.parquet: {len(agg)} configs, {sum(flag)} on the latency frontier")

# ---- dispatcher_f1: per-model routing F1 (scorecard 'quality' for dispatcher) ----
disp = sc[sc["agent"] == "A_dispatcher"].copy()
disp["model"] = disp["config_id"].str.split("__").str[0]
disp["params"] = disp["model"].map(PAR)
g = (disp.groupby("model")
         .agg(params=("params", "first"), f1=("quality", "mean"), n=("quality", "size"))
         .sort_values("params"))
g.to_parquet(str(_DATA / "dispatcher_f1.parquet"))
print(f"dispatcher_f1.parquet: {len(g)} models")

# ---- proxy_vs_quality: param-proxy vs quality-optimal specialist at matched latency
perf_f = perf[perf["peak_mem_gb"] <= _P.MEM_BUDGET_GB]
lg = {r.config_id: r.lat384 for r in perf_f.itertuples()}
modelof = {r.config_id: r.model for r in perf_f.itertuples()}
feas = set(perf_f["config_id"])
specq = {}  # config -> specialist-mean quality
for r in spec.itertuples():
    if r.config_id in feas:
        specq.setdefault(r.config_id, []).append(r.quality)
specq = {c: np.mean(v) for c, v in specq.items()}
rows = []
for eps in np.arange(0.5, 6.01, 0.1):
    eps = round(eps, 1)
    cands = [c for c in specq if lg.get(c, 9e9) <= eps]
    if not cands:
        continue
    by_param = max(cands, key=lambda c: PAR[modelof[c]])
    by_qual = max(cands, key=lambda c: specq[c])
    rows.append(dict(eps=eps,
                     proxy_model=modelof[by_param], Q_proxy=specq[by_param],
                     quality_model=modelof[by_qual], Q_quality=specq[by_qual]))
pv = pd.DataFrame(rows).drop_duplicates(subset=["proxy_model", "quality_model"], keep="first")
pv.to_csv(str(_DATA / "proxy_vs_quality.csv"), index=False, lineterminator='\n')
gap = (pv["Q_quality"] - pv["Q_proxy"])
print(f"proxy_vs_quality.csv: {len(pv)} budgets, mean gap {gap.mean():.3f}, max {gap.max():.3f}")

# ---- policy_compare: fine per-slot-latency sweep, max-params vs max-quality ----
rows = []
for eps in np.arange(1.0, 12.01, 0.1):
    eps = round(eps, 1)
    cands = [c for c in specq if lg.get(c, 9e9) <= eps]
    if not cands:
        continue
    mp = max(cands, key=lambda c: PAR[modelof[c]])
    mq = max(cands, key=lambda c: specq[c])
    rows.append(dict(eps=eps, q_maxparams=specq[mp], q_maxquality=specq[mq],
                     m_maxparams=modelof[mp], m_maxquality=modelof[mq]))
pc = pd.DataFrame(rows)
pc.to_parquet(str(_DATA / "policy_compare.parquet"))
print(f"policy_compare.parquet: {len(pc)} budget steps")
print("part2 intermediates ready.")
