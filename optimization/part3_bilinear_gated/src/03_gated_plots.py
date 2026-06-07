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
OL=[pe.withStroke(linewidth=2.5,foreground="white")]
OUT=str(_FIGS)
fr=pd.read_csv(str(_P.QUALITY_GATED_FRONTIER))

# ===== PLOT A: Bilinear gated Pareto frontier, k in {1,3,5} =====
fig,ax=plt.subplots(figsize=(9.4,5.6))
kcol={1:"#2E6FB0",3:"#C0392B",5:"#E89A2C"}
for k in [1,3,5]:
    d=fr[fr.k==k].sort_values("bil_lat")
    Q=d["bil_Q"].cummax()
    lw=2.4 if k==3 else 1.6; a=1.0 if k==3 else 0.7
    ax.step(d["bil_lat"],Q,where="post",color=kcol[k],lw=lw,alpha=a,zorder=4 if k==3 else 3,
            label=f"k = {k}"+(" (typical)" if k==3 else ""))
    ax.scatter(d["bil_lat"],d["bil_Q"],s=20,color=kcol[k],alpha=a*0.8,zorder=4)
# annotate the optimal allocation on k=3 plateau (concise, placed in open lower space)
d3=fr[fr.k==3].sort_values("bil_lat")
top=d3.loc[d3["bil_Q"].idxmax()]
ax.annotate("gated optimum (k=3):\nSmolLM2-360M router\n+ Llama-3.2-1B specialist\n+ Mistral-7B synthesiser",
            (top["bil_lat"],top["bil_Q"]),fontsize=7.6,fontweight="bold",color="#7a1f15",
            xytext=(-30,-86),textcoords="offset points",ha="left",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#7a1f15",lw=1.1))
ax.margins(x=0.06,y=0.10)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)")
ax.set_ylabel("gated pipeline quality  $Q_{\\mathrm{gated}}$")
ax.set_title("Bilinear (gated) quality--latency frontier on measured data",fontsize=11)
ax.legend(title="activated specialists",loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(f"{OUT}/q15_bilinear_frontier.pdf"); plt.close()

# ===== PLOT B: bilinear vs additive, TRUE gated quality (k=3) =====
fig,ax=plt.subplots(figsize=(9.4,5.4))
d=fr[fr.k==3].sort_values("eps")
ax.step(d["eps"],d["bil_Q"].cummax(),where="post",color="#1a7a3a",lw=2.4,label="bilinear (gated) optimum",zorder=5)
ax.step(d["eps"],d["add_Q"],where="post",color="#C0392B",lw=2.0,label="additive optimum (scored under true gating)",zorder=4)
ax.fill_between(d["eps"],d["add_Q"],d["bil_Q"].cummax(),step="post",alpha=0.15,color="#1a7a3a",zorder=2)
gap=(d["bil_Q"]-d["add_Q"])
ax.annotate(f"the additive allocation,\nrun through the real router gating,\nloses on average {gap.mean():.2f} quality\n(it ignores recall on the used domains)",
            (d["eps"].iloc[len(d)//2],d["add_Q"].iloc[len(d)//2]),fontsize=7.6,color="#922",fontweight="bold",
            xytext=(14,-40),textcoords="offset points",path_effects=OL,arrowprops=dict(arrowstyle="->",color="#922",lw=1.1))
ax.margins(x=0.04,y=0.10)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s, $k{=}3$)")
ax.set_ylabel("true gated pipeline quality")
ax.set_title("Ignoring router--specialist coupling costs quality at every budget",fontsize=11)
ax.legend(loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(f"{OUT}/q16_bilinear_vs_additive.pdf"); plt.close()

# ===== PLOT C (q14): why gating differs -- activated-domain recall vs global F1 =====
qt=pd.read_parquet(str(_P.QUALITY_TABLE))
dd=qt[(qt.agent=="A_dispatcher")&(qt.output.str.len()>0)&(qt.expected_agents.str.len()>0)].copy()
def _parse(s): return set(x for x in str(s).split("|") if x and x not in ("A_dispatcher","A_synth"))
dd["pred"]=dd["output"].apply(_parse); dd["gold"]=dd["expected_agents"].apply(_parse)
dd["model"]=dd["config_id"].str.split("__").str[0]
acts=["A_succ_legittima","A_succ_testamentaria","A_tutela_minori"]
sc=pd.read_csv(str(_P.SCORECARD))
f1g=sc[sc.agent=="A_dispatcher"].groupby(sc["config_id"].str.split("__").str[0])["quality"].mean()
rows=[]
for mdl in dd["model"].unique():
    md=dd[dd.model==mdl]
    rs=[md[md["gold"].apply(lambda g: d_ in g)]["pred"].apply(lambda p: d_ in p).mean() for d_ in acts]
    rows.append((mdl,_P.PARAMS_B[mdl],np.mean(rs),f1g.get(mdl,0)))
dr=pd.DataFrame(rows,columns=["model","params","recall_act","f1_global"])
fig,ax=plt.subplots(figsize=(9,5.4))
ax.scatter(dr["f1_global"],dr["recall_act"],s=80,c=dr["params"],cmap="viridis",edgecolor="white",lw=0.8,zorder=3)
# per-model label offsets to avoid collisions in the crowded top cluster
_off={"smollm2-360m":(8,10),"qwen2.5-0_5b":(4,-13),"ministral-3b":(-50,2),
      "qwen2.5-3b":(6,6),"smollm2-1_7b":(-8,10),"qwen2.5-1_5b":(-10,-13),
      "llama3.2-1b":(8,-12),"llama3.2-3b":(6,4),"gemma2-2b":(6,6),
      "phi3.5-mini":(6,-11),"mistral-7b":(6,8)}
for _,r in dr.iterrows():
    dx,dy=_off.get(r["model"],(5,3))
    ax.annotate(r["model"],(r["f1_global"],r["recall_act"]),fontsize=7.2,
                xytext=(dx,dy),textcoords="offset points",path_effects=OL,zorder=5)
sm=dr[dr.model=="smollm2-360m"].iloc[0]
ax.scatter([sm["f1_global"]],[sm["recall_act"]],s=240,facecolor="none",edgecolor="#C0392B",lw=2.5,zorder=4)
# red callout placed in the open lower-right area, long arrow to the circled point
ax.annotate("chosen by bilinear MILP:\nlow global F1 but HIGH recall\non the activated domains",
            (sm["f1_global"],sm["recall_act"]),fontsize=8,color="#922",fontweight="bold",
            xytext=(0.52,0.42),textcoords="axes fraction",ha="left",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#922",lw=1.2,connectionstyle="arc3,rad=-0.2"))
ax.plot([0,1],[0,1],ls=":",color="#aaa",lw=1)
ax.set_ylim(-0.02,1.12)  # headroom above the y=1.0 cluster
ax.set_xlabel("global routing F1 (all domains)"); ax.set_ylabel("routing recall on the 3 activated domains")
ax.set_title("Why gating differs: global F1 is not recall on the USED domains",fontsize=10.5)
cb=fig.colorbar(ax.collections[0],ax=ax,fraction=0.04,pad=0.02); cb.set_label("params (B)",fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/q14_bilinear_recall.pdf"); plt.close()
print("written q14_bilinear_recall.pdf, q15_bilinear_frontier.pdf, q16_bilinear_vs_additive.pdf")
