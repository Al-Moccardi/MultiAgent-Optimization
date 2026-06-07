import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
"""
#9 Uncertainty: how stable is the frontier under measurement noise?
We lack repeated raw runs, so we do a sensitivity/Monte-Carlo analysis: perturb
each config's throughput and TTFT by multiplicative lognormal noise (sigma=10%,
a realistic laptop-GPU thermal-drift figure), recompute derived latency, re-solve
the MILP at a grid of eps for a fixed k, and report the distribution of the
max capacity reachable at each eps (median + 10-90 percentile band).
Stated clearly as sensitivity, not a CI from real replicates.
"""
import pandas as pd, numpy as np, pulp, json
rng=np.random.default_rng(0)
p=pd.read_parquet(str(_P.PERF_TABLE))
p["model"]=p["config_id"].str.split("__").str[0]; p["quant"]=p["config_id"].str.split("__").str[1]
p["ctx"]=p["config_id"].str.split("__c").str[1].astype(int)
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
p["params"]=p["model"].map(PAR)
M=6.99; T_DISP=15; T_GEN=384
p=p[p["peak_mem_gb"]<=M].reset_index(drop=True)
configs=list(p["config_id"]); modelof={r.config_id:r.model for r in p.itertuples()}
mem={r.config_id:r.peak_mem_gb for r in p.itertuples()}
ttft0={r.config_id:r.ttft_s for r in p.itertuples()}
thr0={r.config_id:r.throughput_tok_s for r in p.itertuples()}
MODELS=sorted(set(modelof.values()),key=lambda m:PAR[m])
grp={}
for r in p.itertuples(): grp.setdefault((r.model,r.quant),[]).append(r.config_id)
ROLES=["d","s","y"]

def milp(eps,k,ld,lg,tl=4):
    pr=pulp.LpProblem("m",pulp.LpMaximize)
    x={(rl,c):pulp.LpVariable(f"x_{rl}_{c}",cat="Binary") for rl in ROLES for c in configs}
    z={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in configs}
    u={m:pulp.LpVariable(f"u_{m}",cat="Binary") for m in MODELS}
    pr+=pulp.lpSum(PAR[m]*u[m] for m in MODELS)
    for rl in ROLES:
        pr+=pulp.lpSum(x[(rl,c)] for c in configs)==1
        for c in configs: pr+=x[(rl,c)]<=z[c]
    for g,cs in grp.items():
        if len(cs)>1: pr+=pulp.lpSum(z[c] for c in cs)<=1
    pr+=pulp.lpSum(mem[c]*z[c] for c in configs)<=M
    for m in MODELS:
        cms=[c for c in configs if modelof[c]==m]
        pr+=u[m]<=pulp.lpSum(x[(rl,c)] for rl in ROLES for c in cms)
    Ld=pulp.lpSum(ld[c]*x[("d",c)] for c in configs)
    Ls=pulp.lpSum(lg[c]*x[("s",c)] for c in configs)
    Ly=pulp.lpSum(lg[c]*x[("y",c)] for c in configs)
    pr+=Ld+k*Ls+Ly<=eps
    pr.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=tl))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    ch={rl:[c for c in configs if x[(rl,c)].value()>0.5][0] for rl in ROLES}
    return sum(PAR[m] for m in {modelof[ch[rl]] for rl in ROLES})

k=3; SIGMA=0.10; N=25
eps_grid=np.round(np.arange(7.0,22.01,1.5),1)
# baseline (no noise)
ld0={c:ttft0[c]+T_DISP/thr0[c] for c in configs}
lg0={c:ttft0[c]+T_GEN/thr0[c] for c in configs}
base={float(e):milp(e,k,ld0,lg0) for e in eps_grid}
# monte-carlo
samples={float(e):[] for e in eps_grid}
for n in range(N):
    fa=rng.lognormal(0,SIGMA,len(configs)); fb=rng.lognormal(0,SIGMA,len(configs))
    thr={c:thr0[c]*fa[i] for i,c in enumerate(configs)}
    ttf={c:ttft0[c]*fb[i] for i,c in enumerate(configs)}
    ld={c:ttf[c]+T_DISP/thr[c] for c in configs}; lg={c:ttf[c]+T_GEN/thr[c] for c in configs}
    for e in eps_grid:
        v=milp(float(e),k,ld,lg)
        if v is not None: samples[float(e)].append(v)

out=[]
print(f"k={k}, sigma={SIGMA*100:.0f}% throughput+TTFT noise, N={N} draws")
print(f"{'eps':>6} {'base':>6} {'p10':>6} {'p50':>6} {'p90':>6}")
for e in eps_grid:
    s=np.array(samples[float(e)])
    if len(s)==0: continue
    p10,p50,p90=np.percentile(s,[10,50,90])
    out.append(dict(eps=float(e),base=base[float(e)],p10=float(p10),p50=float(p50),p90=float(p90)))
    print(f"{e:>6.1f} {base[float(e)]:>6.1f} {p10:>6.1f} {p50:>6.1f} {p90:>6.1f}")
json.dump(out,open(str(_DATA / "uncertainty_k3.json"),"w"),indent=2)
print("\nsaved uncertainty_k3.json")
