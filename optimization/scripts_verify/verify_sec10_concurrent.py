#!/usr/bin/env python3
"""Verify §10.5 (Fig. fig_seq_vs_conc): quality-aware MILP, full 10-model pool,
sequential chain (L_d + k*L_s + L_y, k=3) versus the EXACT concurrent chain
(L_d + Lambda + L_y with Lambda >= lg_c * x_{s,c} for every specialist config).

Paper claims verified here:
  C1  Budget grid eps in {4,5,6,7,8,9,10,12,14,16,18,20,24,28} (14 points):
      concurrent feasible at all 14/14 (from 4 s); sequential at 11/14 (from 7 s).
  C2  Optimal specialist is a sub-1.3B model at 100% of feasible budgets:
      Llama-3.2-1B at 13/14 (concurrent) and 9/11 (sequential); SmolLM2-360M
      only at the tightest budgets (conc eps=4; seq eps in {7,8}).
  C3  Both frontiers plateau at the same Q ~= 1.5991.
  C4  Concurrent Q >= sequential Q at every budget where both are feasible.
Writes out/fig_seq_vs_conc.pdf for visual comparison with the paper figure.

Run:  python verify_sec10_concurrent.py     (~1-2 min, CBC)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "shared" / "data"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
K, M, W_D, W_Y = 3, 6.99, 0.15, 1.0

sc = pd.read_csv(DATA / "quality_scorecard.csv")
perf = pd.read_parquet(DATA / "perf_table.parquet")
perf["model"] = perf.config_id.str.split("__").str[0]
perf["lat_disp"] = perf.ttft_s + 15 / perf.throughput_tok_s
perf["lat_gen"] = perf.ttft_s + 384 / perf.throughput_tok_s
perf = perf[perf.peak_mem_gb <= M]
mem = {r.config_id: r.peak_mem_gb for r in perf.itertuples()}
ld = {r.config_id: r.lat_disp for r in perf.itertuples()}
lg = {r.config_id: r.lat_gen for r in perf.itertuples()}
modelof = {r.config_id: r.model for r in perf.itertuples()}
feas = set(perf.config_id)

SPECDOMS = [a for a in sc.agent.unique() if a not in ("A_dispatcher", "A_synth")]
Qy = {r.config_id: r.quality for r in sc[sc.agent == "A_synth"].itertuples()
      if r.config_id in feas}
F1 = {r.config_id: r.quality for r in sc[sc.agent == "A_dispatcher"].itertuples()
      if r.config_id in feas}
dispcfgs = sorted(F1)
syncfgs = sorted(Qy)
Qspec = {}
for r in sc[sc.agent.isin(SPECDOMS)].itertuples():
    if r.config_id in feas:
        Qspec[(r.agent, r.config_id)] = r.quality
speccfgs = sorted({c for (_, c) in Qspec})
# per-config specialist quality, mean over the domains where the config is
# scored (the part2 canonical aggregation -- no zero-filling)
Qspec_mean = {c: np.mean([Qspec[(d, c)] for d in SPECDOMS if (d, c) in Qspec])
              for c in speccfgs}


def solve(eps, chain):
    pr = pulp.LpProblem("s", pulp.LpMaximize)
    xd = {c: pulp.LpVariable(f"d{c}", cat="Binary") for c in dispcfgs}
    xs = {c: pulp.LpVariable(f"s{c}", cat="Binary") for c in speccfgs}
    xy = {c: pulp.LpVariable(f"y{c}", cat="Binary") for c in syncfgs}
    z = {c: pulp.LpVariable(f"z{c}", cat="Binary") for c in feas}
    pr += pulp.lpSum(xd.values()) == 1
    pr += pulp.lpSum(xs.values()) == 1
    pr += pulp.lpSum(xy.values()) == 1
    for c in dispcfgs: pr += xd[c] <= z[c]
    for c in speccfgs: pr += xs[c] <= z[c]
    for c in syncfgs:  pr += xy[c] <= z[c]
    grp = {}
    for c in feas:
        grp.setdefault(tuple(c.split("__")[:2]), []).append(c)
    for g, cs in grp.items():
        if len(cs) > 1:
            pr += pulp.lpSum(z[c] for c in cs) <= 1
    pr += pulp.lpSum(mem[c] * z[c] for c in feas) <= M
    if chain == "seq":
        pr += (pulp.lpSum(ld[c] * xd[c] for c in dispcfgs)
               + K * pulp.lpSum(lg[c] * xs[c] for c in speccfgs)
               + pulp.lpSum(lg[c] * xy[c] for c in syncfgs)) <= eps
    else:  # exact concurrent: L_d + Lambda + L_y, Lambda >= lg_c * xs_c
        lam = pulp.LpVariable("Lam", lowBound=0)
        for c in speccfgs:
            pr += lam >= lg[c] * xs[c]
        pr += (pulp.lpSum(ld[c] * xd[c] for c in dispcfgs) + lam
               + pulp.lpSum(lg[c] * xy[c] for c in syncfgs)) <= eps
    pr += (W_D * pulp.lpSum(F1[c] * xd[c] for c in dispcfgs)
           + pulp.lpSum(Qspec_mean[c] * xs[c] for c in speccfgs)
           + W_Y * pulp.lpSum(Qy[c] * xy[c] for c in syncfgs))
    pr.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[pr.status] != "Optimal":
        return None
    pick = lambda xv: [c for c, v in xv.items() if v.value() > 0.5][0]
    return pulp.value(pr.objective), pick(xd), pick(xs), pick(xy)


GRID = [4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28]
res = {ch: {} for ch in ("seq", "conc")}
for ch in ("seq", "conc"):
    for e in GRID:
        r = solve(e, ch)
        if r:
            res[ch][e] = r
            print(f"  {ch:4s} eps={e:>3}  Q={r[0]:.4f}  spec={modelof[r[2]]}")
        else:
            print(f"  {ch:4s} eps={e:>3}  infeasible")

nc, ns = len(res["conc"]), len(res["seq"])
print(f"[C1] feasible: concurrent {nc}/14 from {min(res['conc'])} s "
      f"(expect 14/14 from 4); sequential {ns}/14 from {min(res['seq'])} s "
      f"(expect 11/14 from 7)")
sm = {ch: {e: modelof[r[2]] for e, r in res[ch].items()} for ch in res}
cl = sum(1 for m in sm["conc"].values() if m == "llama3.2-1b")
sl = sum(1 for m in sm["seq"].values() if m == "llama3.2-1b")
osm = {ch: sorted(e for e, m in sm[ch].items() if m != "llama3.2-1b") for ch in sm}
print(f"[C2] llama3.2-1b specialist: conc {cl}/{nc} (expect 13/14, "
      f"smollm at {osm['conc']}, expect [4]); seq {sl}/{ns} (expect 9/11, "
      f"smollm at {osm['seq']}, expect [7, 8])")
pc = max(r[0] for r in res["conc"].values())
ps = max(r[0] for r in res["seq"].values())
print(f"[C3] plateau: conc {pc:.4f}  seq {ps:.4f}  (expect both 1.5991)")
dom = all(res["conc"][e][0] >= res["seq"][e][0] - 1e-9 for e in res["seq"])
print(f"[C4] concurrent >= sequential at every shared budget: {dom} (expect True)")

fig, ax = plt.subplots(figsize=(8, 4.5))
for ch, c, lbl in (("seq", "#d62728", "sequential ($L_d+kL_s+L_y$)"),
                   ("conc", "#1f77b4", "concurrent ($L_d+\\Lambda+L_y$)")):
    es = sorted(res[ch])
    ax.plot(es, [res[ch][e][0] for e in es], "o-", color=c, label=lbl)
for ch, dy in (("seq", -0.05), ("conc", 0.03)):
    for e, m in sm[ch].items():
        if m != "llama3.2-1b":
            ax.annotate("360M", (e, res[ch][e][0] + dy), fontsize=7, ha="center")
ax.set_xlabel(r"latency budget $\epsilon$ (s)")
ax.set_ylabel("optimal pipeline quality $Q$")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "fig_seq_vs_conc.pdf")
print(f"wrote {OUT/'fig_seq_vs_conc.pdf'}")
print("DONE sec10")
