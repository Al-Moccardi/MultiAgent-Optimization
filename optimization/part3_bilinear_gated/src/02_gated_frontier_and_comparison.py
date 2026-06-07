import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
from lexsolve import lex_refine   # canonical solution selection (see shared/lib/lexsolve.py)
_FIGS, _DATA = _P.part_dirs(__file__)
"""
Bilinear (gated) quality MILP -- FULL frontier + comparisons.
Computes, on MEASURED data, for each (eps, k):
  - bilinear-optimal allocation & gated quality
  - additive-optimal allocation & its TRUE gated quality (re-scored under gating)
This lets us plot a fair Pareto frontier and the bilinear-vs-additive gap.
"""
import pandas as pd, numpy as np, pulp, json

PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}

# ---- measured per-domain routing recall ----
q=pd.read_parquet(str(_P.QUALITY_TABLE))
disp=q[(q.agent=="A_dispatcher")&(q.output.str.len()>0)&(q.expected_agents.str.len()>0)].copy()
def parse(s): return set(x for x in str(s).split("|") if x and x not in ("A_dispatcher","A_synth"))
disp["pred"]=disp["output"].apply(parse); disp["gold"]=disp["expected_agents"].apply(parse)
disp["model"]=disp["config_id"].str.split("__").str[0]
DOMS=sorted({d for g in disp["gold"] for d in g})
recall={}; recall_n={}
for mdl in disp["model"].unique():
    md=disp[disp.model==mdl]
    for dom in DOMS:
        rel=md[md["gold"].apply(lambda g: dom in g)]
        recall[(mdl,dom)]=float(rel["pred"].apply(lambda p: dom in p).mean()) if len(rel) else 0.0
        recall_n[(mdl,dom)]=len(rel)

sc=pd.read_csv(str(_P.SCORECARD))
perf=pd.read_parquet(str(_P.PERF_TABLE))
perf["model"]=perf["config_id"].str.split("__").str[0]
perf["lat_disp"]=perf["ttft_s"]+15/perf["throughput_tok_s"]
perf["lat_gen"]=perf["ttft_s"]+384/perf["throughput_tok_s"]
perf=perf[perf["peak_mem_gb"]<=6.99]
mem={r.config_id:r.peak_mem_gb for r in perf.itertuples()}
ld={r.config_id:r.lat_disp for r in perf.itertuples()}
lg={r.config_id:r.lat_gen for r in perf.itertuples()}
modelof={r.config_id:r.model for r in perf.itertuples()}
feas=set(perf.config_id)
SPECDOMS=[a for a in sc.agent.unique() if a not in ("A_dispatcher","A_synth")]
tmp={}
for r in sc[sc.agent.isin(SPECDOMS)].itertuples():
    if r.config_id in feas: tmp.setdefault((r.config_id.split("__")[0],r.agent),[]).append(r.quality)
Qspec_m={k:np.mean(v) for k,v in tmp.items()}
Qy={r.config_id:r.quality for r in sc[sc.agent=="A_synth"].itertuples() if r.config_id in feas}
F1={r.config_id:r.quality for r in sc[sc.agent=="A_dispatcher"].itertuples() if r.config_id in feas}
dispcfgs=sorted(F1); syncfgs=sorted(Qy); speccfgs=sorted({c for c in feas if any((c.split("__")[0],d) in Qspec_m for d in SPECDOMS)})
DMODELS=sorted(set(modelof[c] for c in dispcfgs)); SMODELS=sorted(set(modelof[c] for c in speccfgs))
W_D=0.15; W_Y=1.0

def gated_quality(dm, sm, sy_cfg, acts):
    """TRUE gated pipeline quality for a model triple."""
    g=W_D*max(F1[c] for c in dispcfgs if modelof[c]==dm)
    g+=W_Y*Qy[sy_cfg]
    for dom in acts:
        g+=recall.get((dm,dom),0.0)*Qspec_m.get((sm,dom),0.0)/len(acts)
    return g

def solve(eps,k,acts,bilinear=True):
    pr=pulp.LpProblem("b",pulp.LpMaximize)
    xd={c:pulp.LpVariable(f"xd_{c}",cat="Binary") for c in dispcfgs}
    xs={c:pulp.LpVariable(f"xs_{c}",cat="Binary") for c in speccfgs}
    xy={c:pulp.LpVariable(f"xy_{c}",cat="Binary") for c in syncfgs}
    z ={c:pulp.LpVariable(f"z_{c}",cat="Binary") for c in feas}
    pr+=pulp.lpSum(xd.values())==1; pr+=pulp.lpSum(xs.values())==1; pr+=pulp.lpSum(xy.values())==1
    for c in dispcfgs: pr+=xd[c]<=z[c]
    for c in speccfgs: pr+=xs[c]<=z[c]
    for c in syncfgs: pr+=xy[c]<=z[c]
    grp={}
    for c in feas: grp.setdefault(tuple(c.split("__")[:2]),[]).append(c)
    for g,cs in grp.items():
        if len(cs)>1: pr+=pulp.lpSum(z[c] for c in cs)<=1
    mem_expr=pulp.lpSum(mem[c]*z[c] for c in feas)
    pr+=mem_expr<=6.99
    lat_expr=pulp.lpSum(ld[c]*xd[c] for c in dispcfgs)+k*pulp.lpSum(lg[c]*xs[c] for c in speccfgs)+pulp.lpSum(lg[c]*xy[c] for c in syncfgs)
    pr+=lat_expr<=eps
    obj=[W_D*pulp.lpSum(F1[c]*xd[c] for c in dispcfgs), W_Y*pulp.lpSum(Qy[c]*xy[c] for c in syncfgs)]
    if not bilinear:
        for dom in acts: obj.append(pulp.lpSum(Qspec_m.get((modelof[c],dom),0.0)*xs[c] for c in speccfgs)/len(acts))
    else:
        yd={dm:pulp.lpSum(xd[c] for c in dispcfgs if modelof[c]==dm) for dm in DMODELS}
        ys={sm:pulp.lpSum(xs[c] for c in speccfgs if modelof[c]==sm) for sm in SMODELS}
        for dom in acts:
            for dm in DMODELS:
                r=recall.get((dm,dom),0.0)
                if r<=0: continue
                for sm in SMODELS:
                    qd=Qspec_m.get((sm,dom),0.0)
                    if qd<=0: continue
                    w=pulp.LpVariable(f"w_{dom}_{dm}_{sm}",lowBound=0,upBound=1)
                    pr+=w<=yd[dm]; pr+=w<=ys[sm]; pr+=w>=yd[dm]+ys[sm]-1
                    obj.append((r*qd/len(acts))*w)
    obj_expr=pulp.lpSum(obj)
    pr+=obj_expr
    pr.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))  # 60s >> hardest exact stage-1 solve (~48s on the reference laptop)
    if pulp.LpStatus[pr.status]!="Optimal": return None
    _dm=modelof[[c for c in dispcfgs if xd[c].value()>.5][0]]
    _sm=modelof[[c for c in speccfgs if xs[c].value()>.5][0]]
    _ym=modelof[[c for c in syncfgs if xy[c].value()>.5][0]]
    _fix=[pulp.lpSum(xd[c] for c in dispcfgs if modelof[c]==_dm)==1,
          pulp.lpSum(xs[c] for c in speccfgs if modelof[c]==_sm)==1,
          pulp.lpSum(xy[c] for c in syncfgs if modelof[c]==_ym)==1]
    lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=60, extra=_fix)  # objective-preserving canonicalization
    cd=[c for c in dispcfgs if xd[c].value()>.5][0]; cs=[c for c in speccfgs if xs[c].value()>.5][0]; cy=[c for c in syncfgs if xy[c].value()>.5][0]
    lat=ld[cd]+k*lg[cs]+lg[cy]; usedmem=sum(mem[c] for c in feas if z[c].value()>.5)
    return dict(disp=modelof[cd],spec=modelof[cs],synth=modelof[cy],synth_c=cy,lat=lat,mem=usedmem,
                gated_Q=gated_quality(modelof[cd],modelof[cs],cy,acts))

domfreq=pd.Series([d for g in disp["gold"] for d in g]).value_counts()

# ===== FRONTIER — EXACT paper configuration: k in {1,3,5}, eps 6..28.5 step 1.5 =====
# This reproduces the paper's 44-point gated frontier (bilinear > additive at 42/44
# budgets; the mean/max gap are computed and printed below and in verify.py — they are
# never hardcoded). Do NOT coarsen the grid: the headline numbers depend on it.
allrows=[]
for k in [1,3,5]:  # paper-exact: k in {1,3,5}
    acts=list(domfreq.index)[:k]
    # paper-exact grid: arange(6,30,1.5) over k in {1,3,5} -> 44 points
    for eps in np.round(np.arange(6,30,1.5),1):
        b=solve(eps,k,acts,True); a=solve(eps,k,acts,False)
        if not b: continue
        row=dict(k=k,eps=eps,bil_Q=b["gated_Q"],bil_lat=b["lat"],bil_mem=b["mem"],
                 bil=f"{b['disp']}|{b['spec']}|{b['synth']}")
        if a:  # additive optimum, re-scored under TRUE gating
            row.update(add_Q=a["gated_Q"],add=f"{a['disp']}|{a['spec']}|{a['synth']}")
        allrows.append(row)
        # incremental write: progress survives interruption
        pd.DataFrame(allrows).to_csv(str(_P.QUALITY_GATED_FRONTIER),index=False, lineterminator='\n')
        print(f"  k={k} eps={eps}: bilQ={row['bil_Q']:.3f}"
              + (f" addQ={row['add_Q']:.3f}" if 'add_Q' in row else "")+f"  [{len(allrows)} pts]", flush=True)
fr=pd.DataFrame(allrows)
fr.to_csv(str(_P.QUALITY_GATED_FRONTIER),index=False, lineterminator='\n')
print("saved gated frontier:",len(fr),"rows  (paper: 44)")
print()
print("=== k=3 frontier (gated quality) ===")
d=fr[fr.k==3]
for _,r in d.iterrows():
    gap=r.get('bil_Q',0)-r.get('add_Q',np.nan)
    print(f"eps={r['eps']:>5} bilQ={r['bil_Q']:.3f} addQ={r.get('add_Q',float('nan')):.3f} gap={gap:+.3f}  {r['bil']}")
g=(fr['bil_Q']-fr['add_Q']).dropna()
print()
print(f"GATED quality: bilinear >= additive by mean {g.mean():.3f}, max {g.max():.3f}, over {len(g)} pts")
print(f"bilinear strictly better in {(g>0.001).sum()}/{len(g)}")
