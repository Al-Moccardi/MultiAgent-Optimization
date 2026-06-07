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
fr=pd.read_csv(str(_P.QUALITY_ADDITIVE_FRONTIER))
fig,ax=plt.subplots(figsize=(9.4,5.6))
kcol={1:"#2E6FB0",3:"#C0392B",5:"#E89A2C",9:"#6A4C93"}
for k in [1,3,5,9]:
    d=fr[fr["k"]==k].sort_values("eps"); Q=d["Q"].cummax()
    lw=2.4 if k==3 else 1.5; a=1.0 if k==3 else 0.65
    ax.step(d["eps"],Q,where="post",color=kcol[k],lw=lw,alpha=a,zorder=4 if k==3 else 3,
            label=f"k = {k}"+(" (typical)" if k==3 else ""))
# annotate the optimal allocation at plateau (placed in open space, lower-centre)
ax.annotate("optimal pipeline:\ndispatcher Mistral-7B (large)\n+ specialist Llama-1B (small)\n+ synthesiser Mistral-7B (large)",
            (17,1.585),fontsize=8,fontweight="bold",color="#7a1f15",
            xytext=(20,-92),textcoords="offset points",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#7a1f15",lw=1.2))
# reference lines for naive fixed-model policies (k=3), COMPUTED from the released data:
# one loaded configuration of the model serves all three roles (load-once memory holds
# trivially); value = W_D*F1 + mean-specialist-quality + W_Y*Q_synth, maximized over
# that model's feasible configs. No number is hardcoded.
def _fixed_policy(model, W_D=0.15, W_Y=1.0):
    sc2=pd.read_csv(str(_P.SCORECARD)); pf2=pd.read_parquet(str(_P.PERF_TABLE))
    pf2=pf2[pf2["peak_mem_gb"]<=_P.MEM_BUDGET_GB]; feas2=set(pf2["config_id"])
    SP2=[a for a in sc2["agent"].unique() if a not in ("A_dispatcher","A_synth")]
    F2={r.config_id:r.quality for r in sc2[sc2.agent=="A_dispatcher"].itertuples() if r.config_id in feas2}
    Y2={r.config_id:r.quality for r in sc2[sc2.agent=="A_synth"].itertuples() if r.config_id in feas2}
    Sm2={}
    for r in sc2[sc2.agent.isin(SP2)].itertuples():
        if r.config_id in feas2: Sm2.setdefault(r.config_id,[]).append(r.quality)
    Sm2={c:float(np.mean(v)) for c,v in Sm2.items()}
    cands=[c for c in F2 if c.startswith(model) and c in Sm2 and c in Y2]
    return max(W_D*F2[c]+Sm2[c]+W_Y*Y2[c] for c in cands)
_qM=_fixed_policy("mistral-7b"); _qL=_fixed_policy("llama3.2-1b")
ax.axhline(_qM,ls=":",color="#888",lw=1.1)
ax.text(5.5,_qM+0.008,f"all-Mistral-7B (capacity policy): {_qM:.2f}",fontsize=7,ha="left",va="bottom",color="#666",path_effects=OL)
ax.axhline(_qL,ls=":",color="#aaa",lw=1.1)
ax.text(5.5,_qL+0.008,f"all-Llama-1B: {_qL:.2f}",fontsize=7,ha="left",va="bottom",color="#888",path_effects=OL)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)")
ax.set_ylabel("measured pipeline quality (weighted F1 + specialist + synth)")
ax.set_title("Quality-aware frontier (measured): the optimum mixes large and small models",fontsize=10.5)
ax.set_xlim(4,40); ax.set_ylim(1.25,1.70)
ax.legend(title="activated specialists",loc="lower right",frameon=True,ncol=2)
fig.tight_layout(); fig.savefig(str(_FIGS / "q9_quality_frontier.pdf")); plt.close()
print("written q9_quality_frontier.pdf")
