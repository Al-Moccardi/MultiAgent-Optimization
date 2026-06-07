import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
from lexsolve import lex_refine   # canonical solution selection (see shared/lib/lexsolve.py)
_FIGS, _DATA = _P.part_dirs(__file__)
"""
Corrected system MILP (v3) -- 3 role-slots, no specialist-ghost gaming.
======================================================================
Roles, not 11 free agents: dispatcher (d), ONE specialist model shared by all
nine specialists (s), synthesiser (y). This is operationally faithful: the nine
specialists do the same kind of work and share one loaded model.

Decision A (shared instance, sequential): latency is a SUM on the critical path.
Decision B (distinct-model capacity): capacity = sum of params over the DISTINCT
models among the three slots (counted once if two slots share a model).

Per-role tokens (#5): dispatcher T_disp=15 (routing), specialist & synth T_gen=384.
Latency chain (sequential, k activated specialists):
    L_sys(k) = L_d(T_disp) + k * L_s(T_gen) + L_y(T_gen)
Parametric over k in {1,3,5,9} (dataset: mean 2.68, median 2, max 6) (#10).

Memory (load-once): a (model,quant) is charged once; if two slots pick the same
(model,quant) it is one instance. one-context-per-(model,quant).

Capacity is a PARAMETER-COUNT proxy, not measured quality (stated). This version
fixes coherence bugs #1-#11; quality objective still awaits the quality table.
"""
import pandas as pd, numpy as np, pulp, json

p = pd.read_parquet(str(_P.PERF_TABLE))
p["model"]=p["config_id"].str.split("__").str[0]
p["quant"]=p["config_id"].str.split("__").str[1]
p["ctx"]=p["config_id"].str.split("__c").str[1].astype(int)
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,
     "smollm2-1_7b":1.7,"gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,
     "ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
p["params"]=p["model"].map(PAR)
M=6.99; T_DISP=15; T_GEN=384
p["lat_disp"]=p["ttft_s"]+T_DISP/p["throughput_tok_s"]
p["lat_gen"] =p["ttft_s"]+T_GEN /p["throughput_tok_s"]
p=p[p["peak_mem_gb"]<=M].reset_index(drop=True)

configs=list(p["config_id"])
mem ={r.config_id:r.peak_mem_gb for r in p.itertuples()}
ldisp={r.config_id:r.lat_disp for r in p.itertuples()}
lgen ={r.config_id:r.lat_gen  for r in p.itertuples()}
modelof={r.config_id:r.model  for r in p.itertuples()}
MODELS=sorted(set(modelof.values()), key=lambda m:PAR[m])
grp={}
for r in p.itertuples():
    grp.setdefault((r.model,r.quant),[]).append(r.config_id)
ROLES=["d","s","y"]   # dispatcher, specialist(shared), synthesiser

def solve(eps, k, tl=8):
    prob=pulp.LpProblem("syscap3",pulp.LpMaximize)
    x={(rl,c):pulp.LpVariable(f"x_{rl}_{c}",cat="Binary") for rl in ROLES for c in configs}
    z={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in configs}     # config loaded
    u={m:pulp.LpVariable(f"u_{m}",cat="Binary") for m in MODELS}      # model used by some role

    # objective: distinct-model capacity over the 3 slots
    obj_expr = pulp.lpSum(PAR[m]*u[m] for m in MODELS)
    prob += obj_expr

    for rl in ROLES:
        prob += pulp.lpSum(x[(rl,c)] for c in configs)==1     # one config per role
        for c in configs:
            prob += x[(rl,c)] <= z[c]                          # use only loaded
    for g,cs in grp.items():
        if len(cs)>1:
            prob += pulp.lpSum(z[c] for c in cs) <= 1          # one context per group
    mem_expr = pulp.lpSum(mem[c]*z[c] for c in configs)
    prob += mem_expr <= M      # memory (load-once)
    for m in MODELS:                                           # capacity only if a role uses model m
        cms=[c for c in configs if modelof[c]==m]
        prob += u[m] <= pulp.lpSum(x[(rl,c)] for rl in ROLES for c in cms)
    # latency: sequential sum on the critical path
    Ld=pulp.lpSum(ldisp[c]*x[("d",c)] for c in configs)
    Ls=pulp.lpSum(lgen[c]*x[("s",c)] for c in configs)
    Ly=pulp.lpSum(lgen[c]*x[("y",c)] for c in configs)
    prob += Ld + k*Ls + Ly <= eps

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=tl))
    if pulp.LpStatus[prob.status]!="Optimal": return None
    lex_refine(prob, obj_expr, Ld + k*Ls + Ly, mem_expr, tl=tl)  # objective-preserving canonicalization
    chosen={rl:[c for c in configs if x[(rl,c)].value()>0.5][0] for rl in ROLES}
    distinct=sorted({modelof[chosen[rl]] for rl in ROLES}, key=lambda m:PAR[m])
    cap=sum(PAR[m] for m in distinct)
    used_mem=sum(mem[c] for c in configs if z[c].value()>0.5)
    Ls_v=Ld.value()+k*Ls.value()+Ly.value()
    return dict(eps=eps,k=k,capacity=cap,n_distinct=len(distinct),used_mem=used_mem,
                Lsys=Ls_v,Ld=Ld.value(),Ls=Ls.value(),Ly=Ly.value(),
                disp=chosen["d"],spec=chosen["s"],synth=chosen["y"],distinct=distinct)

fast_d=min(ldisp.values()); fast_g=min(lgen.values()); slow_g=max(lgen.values())
print("tokens: disp=%d gen=%d | fastest L_d=%.3f L_g=%.3f | slowest L_g=%.3f\n"%(T_DISP,T_GEN,fast_d,fast_g,slow_g))

allr=[]
for k in [1,3,5,9]:
    emin=fast_d+k*fast_g+fast_g
    emax=fast_d+k*slow_g+slow_g+1
    rows=[]
    for eps in np.round(np.arange(np.floor(emin*10)/10, emax, 1.0),2):
        r=solve(eps,k)
        if r: rows.append(r); allr.append(r)
    fr=pd.DataFrame(rows).sort_values("eps"); prev=None
    print(f"=== k={k} | min eps {emin:.1f}s ===")
    print(f"  {'eps':>6} {'cap(B)':>7} {'#m':>3} {'mem':>6} {'Lsys':>6}  disp | spec | synth")
    for _,r in fr.iterrows():
        if r["capacity"]!=prev:
            print(f"  {r['eps']:>6.1f} {r['capacity']:>7.1f} {r['n_distinct']:>3} {r['used_mem']:>6.2f} {r['Lsys']:>6.1f}  "
                  f"{r['disp'].split('__')[0]} | {r['spec'].split('__')[0]} | {r['synth'].split('__')[0]}")
            prev=r["capacity"]
    print()

df=pd.DataFrame([{k:v for k,v in r.items() if k!='distinct'} for r in allr])
df.to_csv(str(_P.PERF_FRONTIER),index=False, lineterminator='\n')
print("saved syscap3_frontier.csv (%d rows)"%len(df))
