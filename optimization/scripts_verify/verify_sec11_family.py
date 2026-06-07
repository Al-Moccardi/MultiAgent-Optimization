#!/usr/bin/env python3
"""Verify §11.1 (single-family ablation, Figs. fig_family_quality /
fig_family_sweep): quality-aware MILP restricted to the Qwen2.5 pool, exact
concurrent chain (L_d + Lambda + L_y) with per-role context minimums
(dispatcher >= 2048, specialist >= 4096, synthesiser >= 8192).

Paper claims verified here:
  C1  Within-family pooled specialist quality is monotone in size for Qwen
      (0.493 -> 0.679 -> 0.716 for 0.5B/1.5B/3B) but the cross-family pattern
      is non-monotone: llama3.2-1b (0.754) > llama3.2-3b (0.712);
      smollm2-360m (0.659) > smollm2-1_7b (0.603).
  C2  Sweeping eps in [5, 18] step 0.5: optimal specialist = Qwen2.5-1.5B for
      eps < 11 s, switching to the largest Qwen2.5-3B at eps = 11.0 s.
  C3  In the tight regime the allocation gives the dispatcher the 3B model
      while specialist AND synthesiser share one 1.5B load (the paper's
      memory-sharing mechanism); the pattern holds on a non-empty eps range
      below the 11.0 s switch (here 6.5-8.5 s).
  C4  The frontier plateaus at Q ~= 1.5112.
Writes out/fig_family_sweep.pdf and out/fig_family_quality.pdf.

Run:  python verify_sec11_family.py     (~1 min, CBC)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pulp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared" / "lib"))
from lexsolve import lex_refine
DATA = ROOT / "shared" / "data"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)
M, W_D, W_Y = 6.99, 0.15, 1.0
CMIN = {"d": 2048, "s": 4096, "y": 8192}

sc = pd.read_csv(DATA / "quality_scorecard.csv")
perf = pd.read_parquet(DATA / "perf_table.parquet")
perf["model"] = perf.config_id.str.split("__").str[0]
perf["ctx"] = perf.config_id.str.split("__").str[2].str.lstrip("c").astype(int)
perf["lat_disp"] = perf.ttft_s + 15 / perf.throughput_tok_s
perf["lat_gen"] = perf.ttft_s + 384 / perf.throughput_tok_s
perf = perf[perf.peak_mem_gb <= M]

# C1 ------------------------------------------------- family quality structure
SPECA = [a for a in sc.agent.unique() if a not in ("A_dispatcher", "A_synth")]
# record-level pooled means (the aggregation of fig_family_quality)
qt = pd.read_parquet(DATA / "quality_table.parquet")
qt = qt[qt.quality.notna() & qt.agent.isin(SPECA)].copy()
qt["model"] = qt.config_id.str.split("__").str[0]
fam = qt.groupby("model").quality.mean()
exp = {"qwen2.5-0_5b": .493, "qwen2.5-1_5b": .679, "qwen2.5-3b": .716,
       "smollm2-360m": .659, "smollm2-1_7b": .603,
       "llama3.2-1b": .754, "llama3.2-3b": .712}
print("[C1] pooled specialist quality by model (expected in parentheses):")
for m, e in exp.items():
    print(f"      {m:16s} {fam[m]:.3f}  ({e:.3f})")

# Qwen-only pool -----------------------------------------------------------
qp = perf[perf.model.str.startswith("qwen")]
mem = {r.config_id: r.peak_mem_gb for r in qp.itertuples()}
ld = {r.config_id: r.lat_disp for r in qp.itertuples()}
lg = {r.config_id: r.lat_gen for r in qp.itertuples()}
ctx = {r.config_id: r.ctx for r in qp.itertuples()}
modelof = {r.config_id: r.model for r in qp.itertuples()}
feas = set(qp.config_id)

Qspec = {}
for r in sc[sc.agent.isin(SPECA)].itertuples():
    if r.config_id in feas:
        Qspec[(r.agent, r.config_id)] = r.quality
Qy = {r.config_id: r.quality for r in sc[sc.agent == "A_synth"].itertuples()
      if r.config_id in feas}
F1 = {r.config_id: r.quality for r in sc[sc.agent == "A_dispatcher"].itertuples()
      if r.config_id in feas}
dispcfgs = sorted(c for c in F1 if ctx[c] >= CMIN["d"])
syncfgs = sorted(c for c in Qy if ctx[c] >= CMIN["y"])
speccfgs = sorted({c for (_, c) in Qspec if ctx[c] >= CMIN["s"]})
# per-config specialist quality, mean over scored domains (part2 canonical)
Qspec_mean = {c: np.mean([Qspec[(d, c)] for d in SPECA if (d, c) in Qspec])
              for c in speccfgs}


def solve(eps):
    pr = pulp.LpProblem("f", pulp.LpMaximize)
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
    mem_expr = pulp.lpSum(mem[c] * z[c] for c in feas)
    pr += mem_expr <= M
    lam = pulp.LpVariable("Lam", lowBound=0)
    for c in speccfgs:
        pr += lam >= lg[c] * xs[c]
    lat_expr = (pulp.lpSum(ld[c] * xd[c] for c in dispcfgs) + lam
                + pulp.lpSum(lg[c] * xy[c] for c in syncfgs))
    pr += lat_expr <= eps
    obj_expr = (W_D * pulp.lpSum(F1[c] * xd[c] for c in dispcfgs)
                + pulp.lpSum(Qspec_mean[c] * xs[c] for c in speccfgs)
                + W_Y * pulp.lpSum(Qy[c] * xy[c] for c in syncfgs))
    pr += obj_expr
    pr.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[pr.status] != "Optimal":
        return None
    pick = lambda xv: [c for c, v in xv.items() if v.value() > 0.5][0]
    dm, sm, ym = modelof[pick(xd)], modelof[pick(xs)], modelof[pick(xy)]
    fix = [pulp.lpSum(xd[c] for c in dispcfgs if modelof[c] == dm) == 1,
           pulp.lpSum(xs[c] for c in speccfgs if modelof[c] == sm) == 1,
           pulp.lpSum(xy[c] for c in syncfgs if modelof[c] == ym) == 1]
    lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=60, extra=fix)
    return pulp.value(obj_expr), pick(xd), pick(xs), pick(xy)


EPS = [round(e, 1) for e in np.arange(5, 18.01, 0.5)]
res = {}
for e in EPS:
    r = solve(e)
    if r:
        res[e] = r
        print(f"  eps={e:>5}  Q={r[0]:.4f}  disp={modelof[r[1]]:13s} "
              f"spec={modelof[r[2]]:13s} synth={modelof[r[3]]}")

sw = [e for e in sorted(res) if modelof[res[e][2]] == "qwen2.5-3b"]
small = [e for e in sorted(res) if modelof[res[e][2]] == "qwen2.5-1_5b"]
print(f"[C2] specialist = 1.5B on eps {min(small)}..{max(small)}; "
      f"switches to 3B at eps={min(sw)} (expect 11.0)")
pat = [e for e in sorted(res)
       if modelof[res[e][1]] == "qwen2.5-3b"
       and modelof[res[e][2]] == "qwen2.5-1_5b"
       and modelof[res[e][3]] == "qwen2.5-1_5b"]
shared = [e for e in pat if res[e][2] == res[e][3]]
ex = shared[len(shared)//2] if shared else None
print(f"[C3] 3B-dispatcher + both-roles-1.5B pattern on eps "
      f"{pat[0] if pat else '-'}..{pat[-1] if pat else '-'} (expect 6.5..8.5); "
      f"a single 1.5B load serves spec AND synth on eps "
      f"{shared[0] if shared else '-'}..{shared[-1] if shared else '-'} "
      f"(expect a non-empty sub-range, 7.0..8.5)"
      + (f"; e.g. eps={ex}: {res[ex][2]} ({mem[res[ex][2]]:.2f} GB) under "
         f"disp {res[ex][1]} ({mem[res[ex][1]]:.2f} GB)" if shared else ""))
plat = max(r[0] for r in res.values())
print(f"[C4] plateau Q = {plat:.4f} (expect 1.5112)")

fig, ax = plt.subplots(figsize=(8, 4.5))
es = sorted(res)
ax.plot(es, [res[e][0] for e in es], "o-", color="#1f77b4")
for e in es:
    m = modelof[res[e][2]]
    ax.annotate("3B" if m.endswith("3b") else "1.5B",
                (e, res[e][0] + 0.012), fontsize=7, ha="center",
                color="#d62728" if m.endswith("3b") else "#2ca02c")
ax.axvline(min(sw), ls="--", color=".5")
ax.text(min(sw) + 0.1, min(r[0] for r in res.values()),
        f"specialist switches to 3B at $\\epsilon$={min(sw)}", fontsize=8, color=".35")
ax.set_xlabel(r"latency budget $\epsilon$ (s)")
ax.set_ylabel("optimal pipeline quality $Q$ (Qwen pool)")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(OUT / "fig_family_sweep.pdf")

fams = {"Qwen2.5": ["qwen2.5-0_5b", "qwen2.5-1_5b", "qwen2.5-3b"],
        "SmolLM2": ["smollm2-360m", "smollm2-1_7b"],
        "Llama-3.2": ["llama3.2-1b", "llama3.2-3b"]}
fig, ax = plt.subplots(figsize=(8, 4))
xpos, lbls = 0, []
for fname, ms in fams.items():
    xs = np.arange(xpos, xpos + len(ms))
    ax.bar(xs, [fam[m] for m in ms], 0.7,
           label=fname, alpha=.85)
    lbls += [m.split("-")[-1] for m in ms]
    xpos += len(ms) + 1
ax.set_xticks([0, 1, 2, 4, 5, 7, 8])
ax.set_xticklabels(lbls)
ax.set_ylabel("pooled specialist quality")
ax.legend()
ax.grid(alpha=.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig_family_quality.pdf")
print(f"wrote {OUT/'fig_family_sweep.pdf'} and fig_family_quality.pdf")
print("DONE sec11-family")
