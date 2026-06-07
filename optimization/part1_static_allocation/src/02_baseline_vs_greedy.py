import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
"""
#8 Baseline test: does the 3-slot MILP beat a trivial greedy heuristic?
Heuristic (what an engineer would try without an optimizer):
  "Greedy-by-slot": give each role the LARGEST model (max params) whose config
  keeps the system-latency chain within eps, filling the critical-path slots in
  cost order (synth and specialist are the expensive T_gen slots; dispatcher is
  cheap at T_disp). Concretely: independently pick, per slot, the largest-param
  config such that the *current* chain stays <= eps, processing the cheapest-token
  slot (dispatcher) last. We also try the even simpler 'uniform': the largest
  single model put on all three slots that fits memory and latency.
Compare capacity (distinct-model params) MILP vs heuristics at each (eps,k).
"""
import pandas as pd, numpy as np, json, pulp

p=pd.read_parquet(str(_P.PERF_TABLE))
p["model"]=p["config_id"].str.split("__").str[0]; p["quant"]=p["config_id"].str.split("__").str[1]
p["ctx"]=p["config_id"].str.split("__c").str[1].astype(int)
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
p["params"]=p["model"].map(PAR)
M=6.99; T_DISP=15; T_GEN=384
p["lat_disp"]=p["ttft_s"]+T_DISP/p["throughput_tok_s"]
p["lat_gen"]=p["ttft_s"]+T_GEN/p["throughput_tok_s"]
p=p[p["peak_mem_gb"]<=M].reset_index(drop=True)
configs=list(p["config_id"])
mem={r.config_id:r.peak_mem_gb for r in p.itertuples()}
ldisp={r.config_id:r.lat_disp for r in p.itertuples()}
lgen={r.config_id:r.lat_gen for r in p.itertuples()}
modelof={r.config_id:r.model for r in p.itertuples()}
parof={r.config_id:r.params for r in p.itertuples()}
MODELS=sorted(set(modelof.values()),key=lambda m:PAR[m])
grp={}
for r in p.itertuples(): grp.setdefault((r.model,r.quant),[]).append(r.config_id)
ROLES=["d","s","y"]

# ---------- MILP (same as v3) ----------
def milp(eps,k,tl=8):
    pr=pulp.LpProblem("m",pulp.LpMaximize)
    x={(rl,c):pulp.LpVariable(f"x_{rl}_{c}",cat="Binary") for rl in ROLES for c in configs}
    z={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in configs}
    u={m:pulp.LpVariable(f"u_{m}",cat="Binary") for m in MODELS}
    pr += pulp.lpSum(PAR[m]*u[m] for m in MODELS)
    for rl in ROLES:
        pr += pulp.lpSum(x[(rl,c)] for c in configs)==1
        for c in configs: pr += x[(rl,c)]<=z[c]
    for g,cs in grp.items():
        if len(cs)>1: pr += pulp.lpSum(z[c] for c in cs)<=1
    pr += pulp.lpSum(mem[c]*z[c] for c in configs)<=M
    for m in MODELS:
        cms=[c for c in configs if modelof[c]==m]
        pr += u[m] <= pulp.lpSum(x[(rl,c)] for rl in ROLES for c in cms)
    Ld=pulp.lpSum(ldisp[c]*x[("d",c)] for c in configs)
    Ls=pulp.lpSum(lgen[c]*x[("s",c)] for c in configs)
    Ly=pulp.lpSum(lgen[c]*x[("y",c)] for c in configs)
    pr += Ld+k*Ls+Ly<=eps
    pr.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=tl))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    ch={rl:[c for c in configs if x[(rl,c)].value()>0.5][0] for rl in ROLES}
    dist=sorted({modelof[ch[rl]] for rl in ROLES},key=lambda m:PAR[m])
    return sum(PAR[m] for m in dist)

# ---------- Heuristic 1: greedy-by-slot ----------
# cheapest config (mem) per (model) at smallest ctx, to maximize chance of fitting;
# pick largest-param model per slot s.t. latency chain <= eps and memory <= M.
def best_cfg_for_model(m):  # smallest-mem config of model m
    cms=[c for c in configs if modelof[c]==m]
    return min(cms,key=lambda c:mem[c]) if cms else None

def greedy(eps,k):
    # try to maximize distinct params on 3 slots greedily.
    # latency: Ld(disp,15) + k*Ls(spec,384) + Ly(synth,384) <= eps
    # strategy: synth & spec are the expensive slots; dispatcher cheap.
    # Iterate models large->small for each slot, keep if feasible (mem+lat).
    order=sorted(MODELS,key=lambda m:-PAR[m])
    best=None
    # brute over model choices per slot but using only smallest-mem config (heuristic, not MILP)
    for md in order:
        cd=best_cfg_for_model(md)
        for ms in order:
            cs=best_cfg_for_model(ms)
            for my in order:
                cy=best_cfg_for_model(my)
                # memory load-once over distinct configs used
                used={cd,cs,cy}
                # respect one-context-per-group trivially (smallest ctx each, distinct groups ok)
                mtot=sum(mem[c] for c in used)
                if mtot>M: continue
                lat=ldisp[cd]+k*lgen[cs]+lgen[cy]
                if lat>eps: continue
                cap=sum(PAR[m] for m in {md,ms,my})
                if best is None or cap>best: best=cap
    return best

# ---------- Heuristic 2: uniform (same model on all 3 slots) ----------
def uniform(eps,k):
    best=None
    for m in sorted(MODELS,key=lambda x:-PAR[x]):
        c=best_cfg_for_model(m)
        if mem[c]>M: continue
        lat=ldisp[c]+k*lgen[c]+lgen[c]
        if lat>eps: continue
        best=PAR[m]  # one distinct model -> capacity = its params
        break
    return best

fast_d=min(ldisp.values()); fast_g=min(lgen.values()); slow_g=max(lgen.values())
print(f"{'k':>2} {'eps':>6} {'MILP':>6} {'greedy':>7} {'uniform':>8}  verdict")
rows=[]
wins=0; ties=0; tot=0
for k in [1,3,5,9]:
    emin=fast_d+k*fast_g+fast_g; emax=fast_d+k*slow_g+slow_g+1
    for eps in np.round(np.arange(np.floor(emin*10)/10,emax,2.0),2):
        mi=milp(eps,k); gr=greedy(eps,k); un=uniform(eps,k)
        if mi is None: continue
        tot+=1
        v="tie"
        if gr is None or mi>gr+1e-6: v="MILP>greedy"; wins+=1
        elif abs(mi-gr)<=1e-6: v="="; ties+=1
        rows.append(dict(k=k,eps=float(eps),milp=mi,greedy=gr,uniform=un))
        print(f"{k:>2} {eps:>6.1f} {mi:>6.1f} {str(round(gr,1) if gr else '-'):>7} {str(round(un,1) if un else '-'):>8}  {v}")
print()
print(f"MILP strictly > greedy in {wins}/{tot} points; tie in {ties}/{tot}")
json.dump(rows,open(str(_DATA / "baseline_cmp.json"),"w"),indent=2)
