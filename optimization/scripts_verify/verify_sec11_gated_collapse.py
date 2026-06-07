#!/usr/bin/env python3
"""Verify §11.2 (the coupling result is cross-family): re-solve the gated
bilinear MILP (McCormick, model-level) and its coupling-blind additive
counterpart on the Qwen-only pool, and show the gap collapses.

Definitions are the canonical ones of part3_bilinear_gated/src/01:
  * dispatcher rows  = quality_table rows with agent == A_dispatcher and
    non-empty `output` AND non-empty `expected_agents`;
  * parse(s)         = set of '|'-separated tokens minus {A_dispatcher, A_synth};
  * recall[(m,dom)]  = mean over rows with dom in gold of (dom in pred),
    pooled over all of model m's configurations;
  * model-level "global F1" = MEAN over the model's feasible configurations of
    the scorecard A_dispatcher quality (routing F1);
  * configuration choice canonicalized with shared/lib/lexsolve.lex_refine
    (objective-preserving: model triple fixed, then min latency, then min mem).

Paper claims verified here:
  C1  Within Qwen, the best-global-F1 model (Qwen2.5-3B, F1 = 0.4287 ~ "0.43")
      is ALSO best on activated-domain recall (macro 0.895 / micro 0.887
      ~ "0.89"); in the full pool they disagree: SmolLM2-360M has the lowest
      global F1 (0.3886 ~ "0.39") but near-perfect ACT recall (0.989).
  C2  On the Qwen pool the gated and additive optima select the identical
      dispatcher and specialist at every feasible budget: 33/33 integer
      budgets eps = 8..40 (sequential chain, k = 3) and 37/37 budgets
      eps = 4..40 (concurrent chain with context minimums d>=2048 / s>=4096 /
      y>=8192), with gated-quality gap 0 at every budget.

Run:  python verify_sec11_gated_collapse.py     (~3-5 min, CBC)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pulp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared" / "lib"))
from lexsolve import lex_refine

DATA = ROOT / "shared" / "data"
M, W_D, W_Y, K = 6.99, 0.15, 1.0, 3
ACT = ["A_succ_legittima", "A_succ_testamentaria", "A_tutela_minori"]
CMIN = {"d": 2048, "s": 4096, "y": 8192}

# ------------------------------------------------ canonical dispatcher recall
q = pd.read_parquet(DATA / "quality_table.parquet")
disp = q[(q.agent == "A_dispatcher") & (q.output.str.len() > 0)
         & (q.expected_agents.str.len() > 0)].copy()


def parse(s):
    return set(x for x in str(s).split("|")
               if x and x not in ("A_dispatcher", "A_synth"))


disp["pred"] = disp.output.apply(parse)
disp["gold"] = disp.expected_agents.apply(parse)
disp["model"] = disp.config_id.str.split("__").str[0]
DOMS = sorted({d for g in disp.gold for d in g})
recall = {}
for mdl in disp.model.unique():
    md = disp[disp.model == mdl]
    for dom in DOMS:
        rel = md[md.gold.apply(lambda g: dom in g)]
        recall[(mdl, dom)] = float(rel.pred.apply(lambda p: dom in p).mean()) \
            if len(rel) else 0.0

# ------------------------------------------------------------ scorecard maps
sc = pd.read_csv(DATA / "quality_scorecard.csv")
perf = pd.read_parquet(DATA / "perf_table.parquet")
perf["model"] = perf.config_id.str.split("__").str[0]
perf["ctx"] = perf.config_id.str.split("__").str[2].str.lstrip("c").astype(int)
perf["lat_disp"] = perf.ttft_s + 15 / perf.throughput_tok_s
perf["lat_gen"] = perf.ttft_s + 384 / perf.throughput_tok_s
perf = perf[perf.peak_mem_gb <= M]
mem = {r.config_id: r.peak_mem_gb for r in perf.itertuples()}
ld = {r.config_id: r.lat_disp for r in perf.itertuples()}
lg = {r.config_id: r.lat_gen for r in perf.itertuples()}
ctx = {r.config_id: r.ctx for r in perf.itertuples()}
modelof = {r.config_id: r.model for r in perf.itertuples()}
feas_all = set(perf.config_id)
SPECDOMS = [a for a in sc.agent.unique() if a not in ("A_dispatcher", "A_synth")]
tmp = {}
for r in sc[sc.agent.isin(SPECDOMS)].itertuples():
    if r.config_id in feas_all:
        tmp.setdefault((r.config_id.split("__")[0], r.agent), []).append(r.quality)
Qspec_m = {k: np.mean(v) for k, v in tmp.items()}
Qy = {r.config_id: r.quality for r in sc[sc.agent == "A_synth"].itertuples()
      if r.config_id in feas_all}
F1 = {r.config_id: r.quality for r in sc[sc.agent == "A_dispatcher"].itertuples()
      if r.config_id in feas_all}

# C1 -------------------------------------------------------------- diagnostics
def gf1(m):
    return np.mean([v for c, v in F1.items() if modelof[c] == m])


q3f, smf = gf1("qwen2.5-3b"), gf1("smollm2-360m")
def act_recall(m):
    md = disp[disp.model == m]
    rs, tp, gd = [], 0, 0
    for dom in ACT:
        rel = md[md.gold.apply(lambda g: dom in g)]
        t = int(rel.pred.apply(lambda p: dom in p).sum())
        rs.append(t / len(rel))
        tp += t
        gd += len(rel)
    return np.mean(rs), tp / gd


q3m, q3u = act_recall("qwen2.5-3b")
smm, smu = act_recall("smollm2-360m")
print(f"[C1] qwen2.5-3b: global F1={q3f:.4f} (expect 0.4287), ACT recall "
      f"macro={q3m:.3f} micro={q3u:.3f} (expect 0.895/0.887)")
print(f"     smollm2-360m: global F1={smf:.4f} (expect 0.3886), ACT recall "
      f"macro={smm:.3f} (expect 0.989)")

# ----------------------------------------------- Qwen pool, gated vs additive
feas = {c for c in feas_all if modelof[c].startswith("qwen")}


def cfg_lists(chain):
    dc = sorted(c for c in F1 if c in feas
                and (chain == "seq" or ctx[c] >= CMIN["d"]))
    yc = sorted(c for c in Qy if c in feas
                and (chain == "seq" or ctx[c] >= CMIN["y"]))
    scfg = sorted(c for c in feas
                  if (chain == "seq" or ctx[c] >= CMIN["s"])
                  and any((modelof[c], d) in Qspec_m for d in SPECDOMS))
    return dc, scfg, yc


def solve(eps, chain, bilinear):
    dispcfgs, speccfgs, syncfgs = cfg_lists(chain)
    DM = sorted({modelof[c] for c in dispcfgs})
    SM = sorted({modelof[c] for c in speccfgs})
    pr = pulp.LpProblem("b", pulp.LpMaximize)
    xd = {c: pulp.LpVariable(f"xd_{c}", cat="Binary") for c in dispcfgs}
    xs = {c: pulp.LpVariable(f"xs_{c}", cat="Binary") for c in speccfgs}
    xy = {c: pulp.LpVariable(f"xy_{c}", cat="Binary") for c in syncfgs}
    z = {c: pulp.LpVariable(f"z_{c}", cat="Binary") for c in feas}
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
    if chain == "seq":
        lat_expr = (pulp.lpSum(ld[c] * xd[c] for c in dispcfgs)
                    + K * pulp.lpSum(lg[c] * xs[c] for c in speccfgs)
                    + pulp.lpSum(lg[c] * xy[c] for c in syncfgs))
    else:
        lam = pulp.LpVariable("Lam", lowBound=0)
        for c in speccfgs:
            pr += lam >= lg[c] * xs[c]
        lat_expr = (pulp.lpSum(ld[c] * xd[c] for c in dispcfgs) + lam
                    + pulp.lpSum(lg[c] * xy[c] for c in syncfgs))
    pr += lat_expr <= eps
    obj = [W_D * pulp.lpSum(F1[c] * xd[c] for c in dispcfgs),
           W_Y * pulp.lpSum(Qy[c] * xy[c] for c in syncfgs)]
    if not bilinear:
        for dom in ACT:
            obj.append(pulp.lpSum(Qspec_m.get((modelof[c], dom), 0.0) * xs[c]
                                  for c in speccfgs) / len(ACT))
    else:
        yd = {dm: pulp.lpSum(xd[c] for c in dispcfgs if modelof[c] == dm)
              for dm in DM}
        ys = {sm: pulp.lpSum(xs[c] for c in speccfgs if modelof[c] == sm)
              for sm in SM}
        for dom in ACT:
            for dm in DM:
                r = recall.get((dm, dom), 0.0)
                if r <= 0:
                    continue
                for sm in SM:
                    qd = Qspec_m.get((sm, dom), 0.0)
                    if qd <= 0:
                        continue
                    w = pulp.LpVariable(f"w_{dom}_{dm}_{sm}", lowBound=0, upBound=1)
                    pr += w <= yd[dm]
                    pr += w <= ys[sm]
                    pr += w >= yd[dm] + ys[sm] - 1
                    obj.append((r * qd / len(ACT)) * w)
    obj_expr = pulp.lpSum(obj)
    pr += obj_expr
    pr.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[pr.status] != "Optimal":
        return None
    pick = lambda xv, cs: [c for c in cs if xv[c].value() > .5][0]
    dm = modelof[pick(xd, dispcfgs)]
    sm = modelof[pick(xs, speccfgs)]
    ym = modelof[pick(xy, syncfgs)]
    fix = [pulp.lpSum(xd[c] for c in dispcfgs if modelof[c] == dm) == 1,
           pulp.lpSum(xs[c] for c in speccfgs if modelof[c] == sm) == 1,
           pulp.lpSum(xy[c] for c in syncfgs if modelof[c] == ym) == 1]
    lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=60, extra=fix)
    return dict(dm=dm, sm=sm, ym=ym,
                cd=pick(xd, dispcfgs), cs=pick(xs, speccfgs),
                cy=pick(xy, syncfgs))


def gated_quality(r):
    g = W_D * max(F1[c] for c in F1 if modelof[c] == r["dm"] and c in feas)
    g += W_Y * Qy[r["cy"]]
    for dom in ACT:
        g += recall.get((r["dm"], dom), 0.0) * \
            Qspec_m.get((r["sm"], dom), 0.0) / len(ACT)
    return g


for chain, grid, expect in (("seq", range(8, 41), "33/33"),
                            ("conc", range(4, 41), "37/37")):
    same, tot, gapmax = 0, 0, 0.0
    for e in grid:
        g = solve(e, chain, True)
        a = solve(e, chain, False)
        if g is None and a is None:
            continue
        tot += 1
        if g and a and (g["dm"], g["sm"]) == (a["dm"], a["sm"]):
            same += 1
        if g and a:
            gapmax = max(gapmax, abs(gated_quality(g) - gated_quality(a)))
    print(f"[C2] {chain}: identical dispatcher+specialist at {same}/{tot} "
          f"feasible budgets (expect {expect}); max |gated-quality gap| = "
          f"{gapmax:.4f} (expect 0.0000)")
print("DONE sec11-gated")
