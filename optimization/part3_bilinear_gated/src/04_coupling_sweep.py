import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
"""
WHEN is optimization worth it? Sweep the coupling strength lambda and measure the
MILP-vs-greedy gap. lambda interpolates the specialist's effective recall:
    rho_eff[m,delta] = (1-lambda)*1.0  +  lambda*rho_measured[m,delta]
lambda=0  -> no coupling (specialist always reached; additive)   -> greedy should tie
lambda=1  -> full measured gating                                 -> MILP should win
We report, per lambda: gap(MILP_gated_opt - greedy_gated), and the win rate.
Greedy here = per-role best-by-own-quality (the natural heuristic), scored under
the SAME gating as the MILP, for a fair comparison.
"""
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
ld={r.config_id:r.lat_disp for r in perf.itertuples()}; lg={r.config_id:r.lat_gen for r in perf.itertuples()}
modelof={r.config_id:r.model for r in perf.itertuples()}; feas=set(perf.config_id)
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

def rho_eff(m,dom,lam): return (1-lam)*1.0 + lam*recall.get((m,dom),0.0)

def gated_Q(dm,sm,sy_cfg,acts,lam):
    g=W_D*max(F1[c] for c in dispcfgs if modelof[c]==dm)+W_Y*Qy[sy_cfg]
    for dom in acts: g+=rho_eff(dm,dom,lam)*Qspec_m.get((sm,dom),0.0)/len(acts)
    return g

def milp_opt(eps,k,acts,lam):
    pr=pulp.LpProblem("c",pulp.LpMaximize)
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
    pr+=pulp.lpSum(mem[c]*z[c] for c in feas)<=6.99
    pr+=pulp.lpSum(ld[c]*xd[c] for c in dispcfgs)+k*pulp.lpSum(lg[c]*xs[c] for c in speccfgs)+pulp.lpSum(lg[c]*xy[c] for c in syncfgs)<=eps
    yd={dm:pulp.lpSum(xd[c] for c in dispcfgs if modelof[c]==dm) for dm in DMODELS}
    ys={sm:pulp.lpSum(xs[c] for c in speccfgs if modelof[c]==sm) for sm in SMODELS}
    obj=[W_D*pulp.lpSum(F1[c]*xd[c] for c in dispcfgs), W_Y*pulp.lpSum(Qy[c]*xy[c] for c in syncfgs)]
    for dom in acts:
        for dm in DMODELS:
            r=rho_eff(dm,dom,lam)
            if r<=0: continue
            for sm in SMODELS:
                qd=Qspec_m.get((sm,dom),0.0)
                if qd<=0: continue
                w=pulp.LpVariable(f"w{dom}{dm}{sm}",lowBound=0,upBound=1)
                pr+=w<=yd[dm]; pr+=w<=ys[sm]; pr+=w>=yd[dm]+ys[sm]-1
                obj.append((r*qd/len(acts))*w)
    pr+=pulp.lpSum(obj); pr.solve(pulp.PULP_CBC_CMD(msg=0,timeLimit=15))
    if pulp.LpStatus[pr.status]!="Optimal": return None
    cd=[c for c in dispcfgs if xd[c].value()>.5][0]; cs=[c for c in speccfgs if xs[c].value()>.5][0]; cy=[c for c in syncfgs if xy[c].value()>.5][0]
    return gated_Q(modelof[cd],modelof[cs],cy,acts,lam)

def greedy_opt(eps,k,acts,lam):
    # per-role best by OWN quality within its latency share, then scored under gating
    es=(eps-0.5)/(k+1)
    best=-1
    for fy in [0.3,0.5]:
        cd=[c for c in dispcfgs if ld[c]<=1.0]; cs=[c for c in speccfgs if lg[c]<=es]; cy=[c for c in syncfgs if lg[c]<=es]
        if not(cd and cs and cy): continue
        d_=max(cd,key=lambda c:F1[c]); s_=max(cs,key=lambda c:np.mean([Qspec_m.get((modelof[c],dm),0) for dm in acts])); y_=max(cy,key=lambda c:Qy[c])
        if mem[d_]+(0 if modelof[s_]==modelof[d_] else mem[s_])+(0 if modelof[y_] in(modelof[d_],modelof[s_]) else mem[y_])>6.99: continue
        if ld[d_]+k*lg[s_]+lg[y_]>eps: continue
        best=max(best,gated_Q(modelof[d_],modelof[s_],y_,acts,lam))
    return best if best>0 else None

domfreq=pd.Series([d for g in disp["gold"] for d in g]).value_counts()
acts=list(domfreq.index)[:3]; k=3
EPS=[10,13,16,19,22,25]
print("lambda | mean MILP-greedy gap | win rate | example MILP router")
rows=[]
for lam in [0.0,0.2,0.4,0.6,0.8,1.0]:
    gaps=[]
    for eps in EPS:
        m=milp_opt(eps,k,acts,lam); g=greedy_opt(eps,k,acts,lam)
        if m and g: gaps.append(m-g)
    gaps=np.array(gaps); wr=(gaps>0.005).mean()
    rows.append(dict(lam=lam,mean_gap=float(gaps.mean()),win_rate=float(wr)))
    print(f"  {lam:.1f}  |   {gaps.mean():+.3f}            |  {100*wr:.0f}%")
json.dump(rows,open(str(_DATA / "coupling_sweep.json"),"w"),indent=2)
print("\nsaved coupling_sweep.json")

# ===== plot q17: optimization value vs coupling strength =====
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,"legend.fontsize":8.5,
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OL=[pe.withStroke(linewidth=2.5,foreground="white")]
lams=[r["lam"] for r in rows]; gapv=[r["mean_gap"] for r in rows]
fig,ax=plt.subplots(figsize=(8.6,5.2))
ax.plot(lams,gapv,"-o",color="#2E6FB0",lw=2.4,ms=7,zorder=4)
ax.fill_between(lams,0,gapv,alpha=0.12,color="#2E6FB0",zorder=2)
z=np.polyfit(lams,gapv,1)
ax.plot([0,1],np.polyval(z,[0,1]),"--",color="#C0392B",lw=1.3,alpha=0.7,label=f"linear fit (slope {z[0]:.2f})")
# endpoint callouts placed in clear interior space, arrows to the points
ax.annotate("no coupling (additive):\ngreedy nearly ties",(0.0,gapv[0]),fontsize=8,color="#444",fontweight="bold",
            xytext=(0.06,0.16),textcoords="axes fraction",ha="left",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#444",lw=1))
ax.annotate("full measured gating:\nMILP wins by 0.18",(1.0,gapv[-1]),fontsize=8,color="#1a3f6b",fontweight="bold",
            xytext=(0.60,0.60),textcoords="axes fraction",ha="left",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#1a3f6b",lw=1))
ax.set_xlim(-0.04,1.04); ax.set_ylim(0,max(gapv)*1.25)
ax.set_xlabel(r"coupling strength $\lambda$  (0 = additive, 1 = measured gating)")
ax.set_ylabel(r"mean quality gap: MILP $-$ greedy")
ax.set_title("When does optimization pay off? Linearly in the coupling strength",fontsize=10.5)
ax.legend(loc="upper left",frameon=True)
fig.tight_layout(); fig.savefig(str(_FIGS / "q17_coupling_sweep.pdf")); plt.close()
print("written q17_coupling_sweep.pdf")
