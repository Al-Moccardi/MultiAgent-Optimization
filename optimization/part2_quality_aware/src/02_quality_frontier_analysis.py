import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,"legend.fontsize":8.5,
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OL=[pe.withStroke(linewidth=2,foreground="white")]
agg=pd.read_parquet(str(_DATA / "agg_frontier.parquet"))
agg["model"]=agg["config_id"].str.split("__").str[0]
SHORT={"smollm2-360m":"SmolLM2-0.36B","qwen2.5-0_5b":"Qwen2.5-0.5B","llama3.2-1b":"Llama3.2-1B",
 "qwen2.5-1_5b":"Qwen2.5-1.5B","smollm2-1_7b":"SmolLM2-1.7B","gemma2-2b":"Gemma2-2B","qwen2.5-3b":"Qwen2.5-3B",
 "llama3.2-3b":"Llama3.2-3B","ministral-3b":"Granite-3B(MoE)","phi3.5-mini":"Phi3.5-3.8B","mistral-7b":"Mistral-7B"}
MODELS=sorted(agg["model"].unique(),key=lambda m:agg[agg["model"]==m]["params"].iloc[0])
cmap=dict(zip(MODELS,plt.cm.viridis(np.linspace(0,0.92,len(MODELS)))))

# ============ PLOT 1: Quality vs Latency, frontier vs dominated ============
fig,ax=plt.subplots(figsize=(9.2,5.6))
dom=agg[~agg["pareto_lat"]]; fro=agg[agg["pareto_lat"]].sort_values("lat384")
# dominated points (faint, colored by model)
for mdl in MODELS:
    d=dom[dom["model"]==mdl]
    if len(d): ax.scatter(d["lat384"],d["quality"],s=38,color=cmap[mdl],alpha=0.45,edgecolor="white",lw=0.4,zorder=2)
# frontier line + points
ax.plot(fro["lat384"],fro["quality"],"-",color="#C0392B",lw=1.6,zorder=4,alpha=0.7)
for mdl in MODELS:
    f=fro[fro["model"]==mdl]
    if len(f): ax.scatter(f["lat384"],f["quality"],s=130,color=cmap[mdl],edgecolor="#C0392B",lw=2.0,zorder=5)
# annotate frontier members: ONE label per model (dedup), placed to avoid overlap
_seen=set()
for _,r in fro.iterrows():
    if r["model"] in _seen: continue
    _seen.add(r["model"])
    # offset up-left for the leftmost cluster, up-right otherwise, to avoid collisions
    dx,dy=(8,8)
    if r["model"]=="smollm2-360m": dx,dy=(8,-16)
    elif r["model"]=="qwen2.5-0_5b": dx,dy=(8,12)
    ax.annotate(SHORT[r["model"]],(r["lat384"],r["quality"]),fontsize=8,fontweight="bold",
                xytext=(dx,dy),textcoords="offset points",path_effects=OL,zorder=6)
# annotate the big dominated models to make the point (placed below their points)
for mdl in ["mistral-7b","phi3.5-mini"]:
    d=agg[agg["model"]==mdl].sort_values("lat384").iloc[0]
    ax.annotate(SHORT[mdl]+"\n(dominated)",(d["lat384"],d["quality"]),fontsize=7.5,color="#555",
                xytext=(0,-26),textcoords="offset points",ha="center",path_effects=OL,zorder=6,style="italic")
ax.margins(x=0.04,y=0.10)  # headroom so no label touches the frame
ax.set_xlabel("latency for a 384-token answer (s)")
ax.set_ylabel("measured quality (mean over generative agents)")
ax.set_title("Quality–latency frontier: only small models survive",fontsize=11)
# legend by model size
from matplotlib.lines import Line2D
leg=[Line2D([],[],marker="o",ls="",mfc=cmap[m],mec="white",ms=8,label=SHORT[m]) for m in MODELS]
leg.append(Line2D([],[],marker="o",ls="",mfc="none",mec="#C0392B",mew=2,ms=11,label="on Pareto frontier"))
ax.legend(handles=leg,loc="lower right",frameon=True,ncol=2,fontsize=7.5)
fig.tight_layout(); fig.savefig(str(_FIGS / "q1_frontier_latency.pdf")); plt.close()

# ============ PLOT 2: Params vs Quality (the key non-correlation) ============
fig,ax=plt.subplots(figsize=(8.6,5.2))
for mdl in MODELS:
    d=agg[agg["model"]==mdl]
    ax.scatter(d["params"],d["quality"],s=60,color=cmap[mdl],alpha=0.7,edgecolor="white",lw=0.5,zorder=3)
# mean quality per model
g=agg.groupby("model").agg(params=("params","first"),quality=("quality","mean")).sort_values("params")
ax.plot(g["params"],g["quality"],"-o",color="#333",lw=1.4,ms=5,zorder=4,label="mean quality per model")
# trend line
z=np.polyfit(agg["params"],agg["quality"],1); xs=np.linspace(0.3,7.2,50)
ax.plot(xs,np.polyval(z,xs),"--",color="#C0392B",lw=1.5,alpha=0.7,
        label=f"linear fit (r={agg['params'].corr(agg['quality']):.2f})")
# highlight winner & biggest
best=g["quality"].idxmax()
ax.annotate(f"{SHORT[best]}\nBEST quality",(g.loc[best,"params"],g.loc[best,"quality"]),
            fontsize=8.5,fontweight="bold",color="#1a7a3a",xytext=(10,6),textcoords="offset points",path_effects=OL)
ax.annotate(f"{SHORT['mistral-7b']}\n20x larger, worse",(7.2,g.loc["mistral-7b","quality"]),
            fontsize=8.5,fontweight="bold",color="#922",xytext=(-30,-30),textcoords="offset points",path_effects=OL)
ax.set_xlabel("model parameters (B)"); ax.set_ylabel("measured quality")
ax.set_title(f"Parameters weakly predict quality (r={agg['params'].corr(agg['quality']):.2f}; non-monotone)",fontsize=11)
ax.legend(loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(str(_FIGS / "q2_params_quality.pdf")); plt.close()
print("written q1_frontier_latency.pdf, q2_params_quality.pdf")

# ============ PLOT 3: Validation (A) — max-params vs max-quality policy ============
d=pd.read_parquet(str(_DATA / "policy_compare.parquet"))
fig,ax=plt.subplots(figsize=(9.2,5.4))
ax.step(d["eps"],d["q_maxparams"],where="post",color="#C0392B",lw=2.2,
        label="max-parameters policy (what syscap3 picks)",zorder=4)
ax.step(d["eps"],d["q_maxquality"],where="post",color="#1a7a3a",lw=2.2,
        label="max-quality policy (quality-aware)",zorder=4)
ax.fill_between(d["eps"],d["q_maxparams"],d["q_maxquality"],step="post",alpha=0.15,color="#1a7a3a",zorder=2)
# annotate the model max-params chooses at a few points
for eps in [3.1,5.5,9.7]:
    r=d[d["eps"].round(1)==eps]
    if len(r):
        r=r.iloc[0]
        ax.annotate(r["m_maxparams"],(r["eps"],r["q_maxparams"]),fontsize=7,color="#922",
                    xytext=(0,-14),textcoords="offset points",ha="center",path_effects=OL,style="italic")
ax.annotate("Llama3.2-1B (always optimal)",(8,0.754),fontsize=8,color="#1a7a3a",fontweight="bold",
            xytext=(0,6),textcoords="offset points",ha="center",path_effects=OL)
ax.set_xlabel(r"latency budget for the slot (s)")
ax.set_ylabel("resulting measured quality")
ax.set_title("Validation: maximizing parameters wastes budget at every latency",fontsize=11)
ax.legend(loc="lower right",frameon=True); ax.set_ylim(0.55,0.78)
fig.tight_layout(); fig.savefig(str(_FIGS / "q3_policy_compare.pdf")); plt.close()

# ============ PLOT 4: per-domain heterogeneity — best model differs by domain ============
m=pd.read_parquet(str(_DATA / "quality_cost_merged.parquet"))
gen=m[(m["agent"]!="A_dispatcher")&(m["agent"]!="A_synth")].copy()
gen["dom"]=gen["agent"].str.replace("A_","",regex=False)
# best model per domain (by mean quality)
bestrows=[]
for dom,g in gen.groupby("dom"):
    gg=g.groupby("model")["quality"].mean()
    bestrows.append(dict(domain=dom,best_model=gg.idxmax(),best_q=gg.max(),
                         mistral_q=gg.get("mistral-7b",np.nan)))
bd=pd.DataFrame(bestrows).sort_values("best_q",ascending=False)
fig,ax=plt.subplots(figsize=(9.6,5.2))
y=np.arange(len(bd))
ax.barh(y,bd["best_q"],height=0.6,color="#1a7a3a",alpha=0.85,zorder=3,label="best model for that domain")
ax.barh(y,bd["mistral_q"],height=0.28,color="#C0392B",alpha=0.9,zorder=4,label="Mistral-7B (largest)")
for yi,(_,r) in zip(y,bd.iterrows()):
    ax.text(r["best_q"]+0.005,yi,f"{SHORT.get(r['best_model'],r['best_model'])}",va="center",fontsize=7.5,fontweight="bold",path_effects=OL)
ax.set_yticks(y); ax.set_yticklabels([d[:26] for d in bd["domain"]],fontsize=8)
ax.set_xlabel("measured quality"); ax.set_xlim(0,0.92)
ax.set_title("Heterogeneity: the best model differs by legal domain",fontsize=11)
ax.legend(loc="lower right",frameon=True); ax.invert_yaxis()
fig.tight_layout(); fig.savefig(str(_FIGS / "q4_per_domain.pdf")); plt.close()
print("written q3_policy_compare.pdf, q4_per_domain.pdf")
