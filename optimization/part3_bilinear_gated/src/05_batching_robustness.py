import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
from lexsolve import lex_refine   # canonical solution selection (see shared/lib/lexsolve.py)
_FIGS, _DATA = _P.part_dirs(__file__)
"""
Batching-robustness sweep. Latency model with serialization factor sigma in [1/k, 1]:
    L_sys(k,sigma) = L_d + (1 + sigma*(k-1))*L_s + L_y
sigma=1   -> sequential (worst case, current model)
sigma=1/k -> ideal batching (homogeneous shared specialist costs like one call)
sigma enters ONLY the latency constraint; memory (load-once) and quality unchanged.

We re-solve the quality-aware additive MILP across the eps grid for several sigma
and report: (a) the quality-latency frontier per sigma, (b) whether the structural
optimum (mixed: large dispatcher/synth, small specialist) is invariant.
"""
import pandas as pd, numpy as np, pulp, json

PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
sc=pd.read_csv(str(_P.SCORECARD))
perf=pd.read_parquet(str(_P.PERF_TABLE))
perf["model"]=perf["config_id"].str.split("__").str[0]
perf["lat_disp"]=perf["ttft_s"]+15/perf["throughput_tok_s"]
perf["lat_gen"]=perf["ttft_s"]+384/perf["throughput_tok_s"]
perf=perf[perf["peak_mem_gb"]<=6.99]
mem={r.config_id:r.peak_mem_gb for r in perf.itertuples()}
ld={r.config_id:r.lat_disp for r in perf.itertuples()}; lg={r.config_id:r.lat_gen for r in perf.itertuples()}
modelof={r.config_id:r.model for r in perf.itertuples()}; feas=set(perf.config_id)
SPECDOMS=[a for a in sc.agent.unique() if a not in ("A_dispatcher","A_synth")]
tmp={}
for r in sc[sc.agent.isin(SPECDOMS)].itertuples():
    if r.config_id in feas: tmp.setdefault((r.config_id.split("__")[0],r.agent),[]).append(r.quality)
Qspec_m={k:np.mean(v) for k,v in tmp.items()}
Qy={r.config_id:r.quality for r in sc[sc.agent=="A_synth"].itertuples() if r.config_id in feas}
F1={r.config_id:r.quality for r in sc[sc.agent=="A_dispatcher"].itertuples() if r.config_id in feas}
dispcfgs=sorted(F1); syncfgs=sorted(Qy)
speccfgs=sorted({c for c in feas if any((c.split("__")[0],d) in Qspec_m for d in SPECDOMS)})
W_D=0.15; W_Y=1.0
import statistics
acts=SPECDOMS  # additive uses mean specialist quality over all domains
Qspec_mean={c: statistics.mean([Qspec_m.get((modelof[c],d),0.0) for d in SPECDOMS]) for c in speccfgs}

def solve(eps,k,sigma):
    # latency multiplier on the specialist term
    mult = 1.0 + sigma*(k-1)
    pr=pulp.LpProblem("s",pulp.LpMaximize)
    xd={c:pulp.LpVariable(f"d{c}",cat="Binary") for c in dispcfgs}
    xs={c:pulp.LpVariable(f"s{c}",cat="Binary") for c in speccfgs}
    xy={c:pulp.LpVariable(f"y{c}",cat="Binary") for c in syncfgs}
    z ={c:pulp.LpVariable(f"z{c}",cat="Binary") for c in feas}
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
    # LATENCY with serialization factor sigma (only change vs baseline)
    lat_expr=pulp.lpSum(ld[c]*xd[c] for c in dispcfgs)+mult*pulp.lpSum(lg[c]*xs[c] for c in speccfgs)+pulp.lpSum(lg[c]*xy[c] for c in syncfgs)
    pr+=lat_expr<=eps
    obj_expr=W_D*pulp.lpSum(F1[c]*xd[c] for c in dispcfgs)+pulp.lpSum(Qspec_mean[c]*xs[c] for c in speccfgs)+W_Y*pulp.lpSum(Qy[c]*xy[c] for c in syncfgs)
    pr+=obj_expr
    pr.solve(pulp.PULP_CBC_CMD(msg=0))   # exact
    if pulp.LpStatus[pr.status]!="Optimal": return None
    lex_refine(pr, obj_expr, lat_expr, mem_expr)  # objective-preserving canonicalization
    cd=[c for c in dispcfgs if xd[c].value()>.5][0]; cs=[c for c in speccfgs if xs[c].value()>.5][0]; cy=[c for c in syncfgs if xy[c].value()>.5][0]
    lat=ld[cd]+mult*lg[cs]+lg[cy]
    Q=W_D*F1[cd]+Qspec_mean[cs]+W_Y*Qy[cy]
    return dict(eps=eps,sigma=sigma,Q=Q,lat=lat,disp=modelof[cd],spec=modelof[cs],synth=modelof[cy])

k=3
SIGMAS=[1.0, 0.66, 0.5, 1.0/3.0]  # seq, partial, partial, ideal-batch (1/k)
EPS=np.round(np.arange(6,30.01,1.0),1)
rows=[]
print("sigma | example optima (eps=10/16/22) disp|spec|synth | spec always small?")
for sig in SIGMAS:
    specset=set()
    for eps in EPS:
        r=solve(eps,k,sig)
        if r: rows.append(r); specset.add(r["spec"])
    # sample optima
    samp=[]
    for e in [10,16,22]:
        r=solve(e,k,sig)
        if r: samp.append(f"{r['disp'][:8]}|{r['spec'][:8]}|{r['synth'][:8]}")
    small = all(PAR[s]<=1.2 for s in specset)
    print(f" {sig:.2f} | {samp} | {'YES' if small else 'NO: '+str(sorted(specset))}")
df=pd.DataFrame(rows); df.to_csv(str(_DATA / "sigma_sweep.csv"),index=False, lineterminator='\n')
print(f"\nsaved sigma_sweep.csv ({len(df)} rows)")
# robustness summary: specialist params chosen, per sigma
print("\n=== specialist model chosen across all eps, per sigma ===")
for sig in SIGMAS:
    s=df[df.sigma==sig]
    vc=s["spec"].value_counts().to_dict()
    print(f" sigma={sig:.2f}: {vc}")
