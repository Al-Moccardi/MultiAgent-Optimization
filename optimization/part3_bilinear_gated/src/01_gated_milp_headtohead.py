import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
from lexsolve import lex_refine   # canonical solution selection (see shared/lib/lexsolve.py)
_FIGS, _DATA = _P.part_dirs(__file__)
"""BILINEAR quality MILP, efficient: McCormick at MODEL level (not config)."""
import pandas as pd, numpy as np, pulp, json
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
q=pd.read_parquet(str(_P.QUALITY_TABLE))
disp=q[(q.agent=="A_dispatcher")&(q.output.str.len()>0)&(q.expected_agents.str.len()>0)].copy()
def parse(s): return set(x for x in str(s).split("|") if x and x not in ("A_dispatcher","A_synth"))
disp["pred"]=disp["output"].apply(parse); disp["gold"]=disp["expected_agents"].apply(parse)
disp["model"]=disp["config_id"].str.split("__").str[0]
DOMS=sorted({d for g in disp["gold"] for d in g})
recall={}
for mdl in disp["model"].unique():
    md=disp[disp.model==mdl]
    for dom in DOMS:
        rel=md[md["gold"].apply(lambda g: dom in g)]
        recall[(mdl,dom)]=float(rel["pred"].apply(lambda p: dom in p).mean()) if len(rel) else 0.0

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
# best quality per (MODEL, domain) and per model for synth/dispatcher (max over configs of that model)
Qspec_m={}  # (model,domain)->quality (mean over that model's configs)
tmp={}
for r in sc[sc.agent.isin(SPECDOMS)].itertuples():
    if r.config_id in feas: tmp.setdefault((r.config_id.split("__")[0],r.agent),[]).append(r.quality)
for k,v in tmp.items(): Qspec_m[k]=np.mean(v)
Qy={r.config_id:r.quality for r in sc[sc.agent=="A_synth"].itertuples() if r.config_id in feas}
F1={r.config_id:r.quality for r in sc[sc.agent=="A_dispatcher"].itertuples() if r.config_id in feas}
dispcfgs=sorted(F1); syncfgs=sorted(Qy); speccfgs=sorted({c for c in feas if (c.split("__")[0],SPECDOMS[0]) in Qspec_m or any((c.split("__")[0],d) in Qspec_m for d in SPECDOMS)})
DMODELS=sorted(set(modelof[c] for c in dispcfgs))
SMODELS=sorted(set(modelof[c] for c in speccfgs))
W_D=0.15; W_Y=1.0

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
        for dom in acts:
            obj.append(pulp.lpSum(Qspec_m.get((modelof[c],dom),0.0)*xs[c] for c in speccfgs)/len(acts))
    else:
        # model-level indicators: yd[dm]=1 if chosen dispatcher is model dm; ys[sm] similarly
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
    pr.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    _dm=modelof[[c for c in dispcfgs if xd[c].value()>.5][0]]
    _sm=modelof[[c for c in speccfgs if xs[c].value()>.5][0]]
    _ym=modelof[[c for c in syncfgs if xy[c].value()>.5][0]]
    _fix=[pulp.lpSum(xd[c] for c in dispcfgs if modelof[c]==_dm)==1,
          pulp.lpSum(xs[c] for c in speccfgs if modelof[c]==_sm)==1,
          pulp.lpSum(xy[c] for c in syncfgs if modelof[c]==_ym)==1]
    lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=60, extra=_fix)  # objective-preserving canonicalization
    cd=[c for c in dispcfgs if xd[c].value()>.5][0]; cs=[c for c in speccfgs if xs[c].value()>.5][0]; cy=[c for c in syncfgs if xy[c].value()>.5][0]
    return dict(eps=eps,Q=pulp.value(pr.objective),disp=modelof[cd],spec=modelof[cs],synth=modelof[cy])

domfreq=pd.Series([d for g in disp["gold"] for d in g]).value_counts()
acts=list(domfreq.index)[:3]
print("activated domains:", [a.replace('A_','') for a in acts])
print(f"\n{'eps':>4}  ADDITIVE                          BILINEAR                          differ?")
rows=[]
for eps in [8,10,12,14,16,18,20,24]:
    a=solve(eps,3,acts,False); b=solve(eps,3,acts,True)
    if not(a and b): 
        print(f"{eps:>4}  (solver returned None)"); continue
    add=f"{a['disp'][:10]}|{a['spec'][:10]}|{a['synth'][:10]}"
    bil=f"{b['disp'][:10]}|{b['spec'][:10]}|{b['synth'][:10]}"
    diff="<<< DIFFERENT" if (a['disp'],a['spec'],a['synth'])!=(b['disp'],b['spec'],b['synth']) else "same"
    print(f"{eps:>4}  {add:33s} {bil:33s} {diff}")
    rows.append(dict(eps=eps,**{f"add_{r}":a[r] for r in ['disp','spec','synth']},**{f"bil_{r}":b[r] for r in ['disp','spec','synth']},differ=diff))
json.dump(rows,open(str(_DATA / "bilinear_vs_additive.json"),"w"),indent=2)
nd=sum(1 for r in rows if 'DIFF' in r['differ'])
print(f"\nDiverse in {nd}/{len(rows)} budget. Recall di esempio (Mistral): succ_test={recall.get(('mistral-7b','A_succ_testamentaria'),0):.2f}, patr_com={recall.get(('mistral-7b','A_patrimoniale_comunione'),0):.2f}")
