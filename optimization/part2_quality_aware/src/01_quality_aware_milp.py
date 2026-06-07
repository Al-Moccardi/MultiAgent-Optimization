import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
from lexsolve import lex_refine   # canonical solution selection (see shared/lib/lexsolve.py)
_FIGS, _DATA = _P.part_dirs(__file__)
"""
QUALITY-AWARE MILP on MEASURED data (the final step).
Replaces the parameter proxy with measured per-role quality from the scorecard.
Roles: dispatcher (routing F1), one shared specialist model, synthesiser.
Objective: maximize measured pipeline quality
   Q = w_d * F1_dispatcher(c_d) + (1/k) * sum_{j in activated} quality_spec_j(c_s)
       + w_y * quality_synth(c_y)
   (specialists share one model c_s; we average measured per-domain quality over
    the k activated domains for that model.)
Constraints: load-once memory <= M, sequential latency <= eps, one config per role.
Compared against greedy (per-slot best) and the parameter-capacity allocation.
"""
import pandas as pd, numpy as np, pulp, json

sc=pd.read_csv(str(_P.SCORECARD))
perf=pd.read_parquet(str(_P.PERF_TABLE))
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
M=6.99; T_DISP=15; T_GEN=384
perf["model"]=perf["config_id"].str.split("__").str[0]
perf["lat_disp"]=perf["ttft_s"]+T_DISP/perf["throughput_tok_s"]
perf["lat_gen"]=perf["ttft_s"]+T_GEN/perf["throughput_tok_s"]
perf=perf[perf["peak_mem_gb"]<=M]
mem={r.config_id:r.peak_mem_gb for r in perf.itertuples()}
ldisp={r.config_id:r.lat_disp for r in perf.itertuples()}
lgen={r.config_id:r.lat_gen for r in perf.itertuples()}
modelof={r.config_id:r.model for r in perf.itertuples()}
feasible=set(perf["config_id"])

SPEC_DOMAINS=[a for a in sc["agent"].unique() if a not in ("A_dispatcher","A_synth")]
# quality lookups
F1d={(r.config_id):r.quality for r in sc[sc["agent"]=="A_dispatcher"].itertuples() if r.config_id in feasible}
Qsyn={(r.config_id):r.quality for r in sc[sc["agent"]=="A_synth"].itertuples() if r.config_id in feasible}
# specialist quality per (domain, config)
Qspec={}
for r in sc[sc["agent"].isin(SPEC_DOMAINS)].itertuples():
    if r.config_id in feasible: Qspec[(r.agent,r.config_id)]=r.quality
# mean specialist quality of a config over k activated domains (use all available domains as the pool)
def spec_quality(cfg, k, domains):
    vals=[Qspec[(d,cfg)] for d in domains if (d,cfg) in Qspec]
    return np.mean(vals) if vals else None

speccfgs=sorted({c for (_,c) in Qspec})           # configs with specialist quality
syncfgs=sorted(Qsyn); dispcfgs=sorted(F1d)
W_D=0.15; W_Y=1.0   # routing weighted less; synth and specialists full

def solve(eps,k,domains):
    pr=pulp.LpProblem("q",pulp.LpMaximize)
    xd={c:pulp.LpVariable(f"xd_{c}",cat="Binary") for c in dispcfgs}
    xs={c:pulp.LpVariable(f"xs_{c}",cat="Binary") for c in speccfgs}
    xy={c:pulp.LpVariable(f"xy_{c}",cat="Binary") for c in syncfgs}
    z ={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in feasible}
    pr+=pulp.lpSum(xd[c] for c in dispcfgs)==1
    pr+=pulp.lpSum(xs[c] for c in speccfgs)==1
    pr+=pulp.lpSum(xy[c] for c in syncfgs)==1
    for c in dispcfgs: pr+=xd[c]<=z[c]
    for c in speccfgs: pr+=xs[c]<=z[c]
    for c in syncfgs:  pr+=xy[c]<=z[c]
    # one context per (model,quant) group
    grp={}
    for c in feasible:
        g=tuple(c.split("__")[:2]); grp.setdefault(g,[]).append(c)
    for g,cs in grp.items():
        if len(cs)>1: pr+=pulp.lpSum(z[c] for c in cs)<=1
    mem_expr=pulp.lpSum(mem[c]*z[c] for c in feasible)
    pr+=mem_expr<=M
    # latency: disp(15) + k*spec(384) + synth(384)
    lat_expr=(pulp.lpSum(ldisp[c]*xd[c] for c in dispcfgs)
         + k*pulp.lpSum(lgen[c]*xs[c] for c in speccfgs)
         + pulp.lpSum(lgen[c]*xy[c] for c in syncfgs))
    pr+=lat_expr <= eps
    # objective: measured quality
    sq={c:spec_quality(c,k,domains) for c in speccfgs}
    obj_expr=(W_D*pulp.lpSum(F1d[c]*xd[c] for c in dispcfgs)
         + pulp.lpSum((sq[c] or 0)*xs[c] for c in speccfgs)
         + W_Y*pulp.lpSum(Qsyn[c]*xy[c] for c in syncfgs))
    pr+=obj_expr
    pr.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=8))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=8)  # objective-preserving canonicalization
    cd=[c for c in dispcfgs if xd[c].value()>0.5][0]
    cs=[c for c in speccfgs if xs[c].value()>0.5][0]
    cy=[c for c in syncfgs if xy[c].value()>0.5][0]
    Q=W_D*F1d[cd]+ (sq[cs] or 0) + W_Y*Qsyn[cy]
    used=sum(mem[c] for c in feasible if z[c].value()>0.5)
    lat=ldisp[cd]+k*lgen[cs]+lgen[cy]
    return dict(eps=eps,k=k,Q=Q,disp=cd,spec=cs,synth=cy,used_mem=used,lat=lat,
                f1=F1d[cd],qs=sq[cs],qy=Qsyn[cy])

domains=SPEC_DOMAINS
fast=min(ldisp.values())+min(lgen.values())*2
print("QUALITY-AWARE MILP on measured data. domains:",len(domains))
print(f"{'k':>2} {'eps':>5} {'Q':>6} {'mem':>5} {'lat':>5}  disp | spec | synth   (f1/qs/qy)")
allr=[]
for k in [1,3,5,9]:
    prev=None
    for eps in np.round(np.arange(4.0,40.1,0.5),1):
        r=solve(eps,k,domains)
        if r and r["Q"]!=prev:
            allr.append(r); prev=r["Q"]
            print(f"{k:>2} {eps:>5.1f} {r['Q']:>6.3f} {r['used_mem']:>5.2f} {r['lat']:>5.1f}  "
                  f"{r['disp'].split('__')[0][:11]:11s}|{r['spec'].split('__')[0][:11]:11s}|{r['synth'].split('__')[0][:11]:11s} "
                  f"({r['f1']:.2f}/{r['qs']:.2f}/{r['qy']:.2f})")
    print()
pd.DataFrame([{k:v for k,v in r.items()} for r in allr]).to_csv(str(_P.QUALITY_ADDITIVE_FRONTIER),index=False, lineterminator='\n')
print("saved quality_frontier.csv")
