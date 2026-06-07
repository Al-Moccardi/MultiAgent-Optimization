import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
"""
Heterogeneous-specialist MILP (v4).
===================================
The performance-only objective (params) is sortable, so a greedy heuristic matches
the MILP. We make the problem realistic: the 9 specialists cover DIFFERENT legal
domains, and each model has a different AFFINITY for each domain. Now "biggest
model" is no longer optimal per role -- a smaller model may be the best fit for a
given domain -- so the objective is NOT sortable and the MILP wins legitimately.

IMPORTANT HONESTY NOTE: we have no measured affinities yet (they require the
quality table, still running). The affinities here are SYNTHETIC but plausible:
each model has a latent skill ~ increasing-but-noisy in log-params, and each
(model,domain) affinity = base skill + domain-specific deviation. This is a
DEMONSTRATION on synthetic affinities, clearly labelled as such -- not a measured
result. When the quality table lands, these A[m,domain] are replaced by measured
per-domain quality and the same MILP runs unchanged.

Execution model unchanged from v3: shared instance, sequential latency, load-once
memory, per-role tokens (disp 15, gen 384), parametric over k activated specialists.
Capacity/quality objective: sum over the k activated specialist-domains of the
affinity of the chosen specialist model + synth quality (+ small dispatcher term).
"""
import pandas as pd, numpy as np, pulp, json

rng=np.random.default_rng(20240530)
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
MODELS=sorted(set(modelof.values()),key=lambda m:PAR[m])
grp={}
for r in p.itertuples(): grp.setdefault((r.model,r.quant),[]).append(r.config_id)

DOMAINS=["succ_testamentaria","succ_legittima","patrimoniale_comunione","patrimoniale_separazione",
         "separazione_consensuale","separazione_contenziosa","tutela_minori",
         "volontaria_giurisdizione","negoziazione_mediazione"]

# ---- SYNTHETIC affinities A[model, domain] in [0,1], plausible but non-monotone ----
# latent skill: increasing but saturating in log-params, + model-specific noise
# (MoE penalised slightly to reflect its odd behaviour). Domain deviations break monotonicity.
def latent_skill(m):
    base=0.45+0.32*np.log(PAR[m]/0.36)/np.log(7.2/0.36)   # 0.45..0.77 with size
    noise=rng.normal(0,0.05)
    if m=="ministral-3b": base-=0.06                        # MoE quirk
    return np.clip(base+noise,0.05,0.95)
skill={m:latent_skill(m) for m in MODELS}
# domain-specific deviation: each model has strengths/weaknesses across domains
A={}
for m in MODELS:
    devs=rng.normal(0,0.12,len(DOMAINS))   # +/-0.12 swings -> a small model can top a domain
    for j,dom in enumerate(DOMAINS):
        A[(m,dom)]=float(np.clip(skill[m]+devs[j],0.02,0.99))
# synth quality ~ latent skill (no domain); dispatcher routing quality ~ latent skill (small weight)
Qsyn={m:float(np.clip(skill[m]+rng.normal(0,0.04),0.02,0.99)) for m in MODELS}
Qdisp={m:float(np.clip(0.5+0.45*np.log(PAR[m]/0.36)/np.log(7.2/0.36)+rng.normal(0,0.04),0.02,0.99)) for m in MODELS}
W_DISP=0.15  # small weight: routing matters but less than generation quality

json.dump({"A":{f"{m}|{d}":A[(m,d)] for m in MODELS for d in DOMAINS},
           "Qsyn":Qsyn,"Qdisp":Qdisp,"skill":skill,"domains":DOMAINS,
           "note":"SYNTHETIC affinities, plausible non-monotone; replace with measured quality"},
          open(str(_DATA / "affinities_synth.json"),"w"),indent=2)

ROLES=["d"]+[f"s{j}" for j in range(9)]+["y"]   # dispatcher, 9 heterogeneous specialists, synth

def milp(eps,k,tl=12):
    # k activated specialists = the FIRST k domains (any fixed subset; domains symmetric a priori).
    act=[f"s{j}" for j in range(k)]
    pr=pulp.LpProblem("v4",pulp.LpMaximize)
    x={(rl,c):pulp.LpVariable(f"x_{rl}_{c}",cat="Binary") for rl in ROLES for c in configs}
    z={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in configs}
    for rl in ROLES:
        pr+=pulp.lpSum(x[(rl,c)] for c in configs)==1
        for c in configs: pr+=x[(rl,c)]<=z[c]
    for g,cs in grp.items():
        if len(cs)>1: pr+=pulp.lpSum(z[c] for c in cs)<=1
    pr+=pulp.lpSum(mem[c]*z[c] for c in configs)<=M
    # latency: dispatcher(15) + sum over k activated specialists Lgen + synth Lgen
    Ld=pulp.lpSum(ldisp[c]*x[("d",c)] for c in configs)
    Lact=pulp.lpSum(lgen[c]*x[(s,c)] for s in act for c in configs)
    Ly=pulp.lpSum(lgen[c]*x[("y",c)] for c in configs)
    pr+=Ld+Lact+Ly<=eps
    # objective: sum of activated specialists' domain-affinity + synth quality + small dispatcher term
    obj=[]
    for j,s in enumerate(act):
        dom=DOMAINS[j]
        obj+=[A[(modelof[c],dom)]*x[(s,c)] for c in configs]
    obj+=[Qsyn[modelof[c]]*x[("y",c)] for c in configs]
    obj+=[W_DISP*Qdisp[modelof[c]]*x[("d",c)] for c in configs]
    pr+=pulp.lpSum(obj)
    pr.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=tl))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    ch={rl:[c for c in configs if x[(rl,c)].value()>0.5][0] for rl in ROLES}
    q=pulp.value(pr.objective)
    used=sum(mem[c] for c in configs if z[c].value()>0.5)
    return dict(eps=eps,k=k,quality=float(q),used_mem=used,chosen=ch)

# ---- greedy heuristic (the realistic competitor): per-slot, pick the config that
# maximizes that slot's own contribution subject to a per-slot latency share; it
# CANNOT see the joint memory/latency coupling. We give it the SAME latency budget,
# split proportionally, and let it pick best-affinity per domain greedily. ----
def greedy(eps,k):
    act=[f"s{j}" for j in range(k)]
    # latency shares: disp tiny; split remaining between k specialists + synth equally by token weight
    # heuristic: give each gen-slot eps/(k+1) minus disp; pick best config under that share + mem.
    # process slots independently (greedy), track memory load-once.
    loaded={}  # group -> config chosen (smallest ctx of that model/quant)
    def cheapest(m):  # smallest-mem config of model m
        cms=[c for c in configs if modelof[c]==m]; return min(cms,key=lambda c:mem[c]) if cms else None
    # dispatcher: best routing quality whose Ldisp small (it's tiny anyway) -> pick max Qdisp that fits mem alone
    chosen={}
    budget=eps
    # dispatcher
    best=None
    for m in sorted(MODELS,key=lambda x:-Qdisp[x]):
        c=cheapest(m)
        if mem[c]<=M: best=c; break
    chosen["d"]=best; budget-=ldisp[best]
    memused={best:mem[best]}
    share=budget/(k+1)  # equal split among k specialists + synth
    # specialists: each domain picks best-affinity config under its share & remaining mem
    for j,s in enumerate(act):
        dom=DOMAINS[j]; cand=None; cb=-1
        for c in sorted(configs,key=lambda c:-A[(modelof[c],dom)]):
            if lgen[c]>share: continue
            extra=0 if c in memused else mem[c]
            if sum(memused.values())+extra>M: continue
            cand=c; break
        if cand is None:  # fallback: smallest model that fits
            for c in sorted(configs,key=lambda c:mem[c]):
                extra=0 if c in memused else mem[c]
                if lgen[c]<=share and sum(memused.values())+extra<=M: cand=c;break
        if cand is None: return None
        chosen[s]=cand; memused[cand]=mem[cand]
    # synth: best Qsyn under share & mem
    cand=None
    for c in sorted(configs,key=lambda c:-Qsyn[modelof[c]]):
        extra=0 if c in memused else mem[c]
        if lgen[c]<=share and sum(memused.values())+extra<=M: cand=c;break
    if cand is None:
        for c in sorted(configs,key=lambda c:mem[c]):
            extra=0 if c in memused else mem[c]
            if lgen[c]<=share and sum(memused.values())+extra<=M: cand=c;break
    if cand is None: return None
    chosen["y"]=cand
    # compute realized latency & quality; reject if over budget
    Ld=ldisp[chosen["d"]]; Lact=sum(lgen[chosen[s]] for s in act); Ly=lgen[chosen["y"]]
    if Ld+Lact+Ly>eps+1e-6: return None
    q=sum(A[(modelof[chosen[f"s{j}"]],DOMAINS[j])] for j in range(k))+Qsyn[modelof[chosen["y"]]]+W_DISP*Qdisp[modelof[chosen["d"]]]
    return dict(quality=float(q),Lsys=Ld+Lact+Ly)

fast_d=min(ldisp.values()); fast_g=min(lgen.values()); slow_g=max(lgen.values())
print("SYNTHETIC affinities (declared). skill range %.2f-%.2f"%(min(skill.values()),max(skill.values())))
print(f"{'k':>2} {'eps':>6} {'MILP_q':>7} {'greedy_q':>8} {'gap':>6}  winner")
rows=[]; wins=0; tot=0
for k in [1,3,5,9]:
    emin=fast_d+k*fast_g+fast_g; emax=fast_d+k*slow_g+slow_g+1
    for eps in np.round(np.arange(np.floor(emin*10)/10,emax,2.0),2):
        mi=milp(eps,k); gr=greedy(eps,k)
        if mi is None: continue
        tot+=1
        gq=gr["quality"] if gr else None
        gap=(mi["quality"]-gq) if gq is not None else None
        w="MILP" if (gq is None or mi["quality"]>gq+1e-6) else "tie"
        if w=="MILP": wins+=1
        rows.append(dict(k=k,eps=float(eps),milp_q=mi["quality"],greedy_q=gq))
        print(f"{k:>2} {eps:>6.1f} {mi['quality']:>7.3f} {(round(gq,3) if gq else '-'):>8} {(round(gap,3) if gap is not None else '-'):>6}  {w}")
print()
print(f"MILP strictly > greedy in {wins}/{tot} points ({100*wins/tot:.0f}%)")
json.dump(rows,open(str(_DATA / "baseline_v4.json"),"w"),indent=2)
