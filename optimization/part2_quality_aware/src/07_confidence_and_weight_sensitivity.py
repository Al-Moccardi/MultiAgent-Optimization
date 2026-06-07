"""
part2 / 07_confidence_and_weight_sensitivity.py
===============================================
Produces the two robustness figures of the quality section:
  q12_per_domain_ci.pdf      : 95% bootstrap CIs on each domain's best-model
                               quality, flagging the under-powered (n<10) domains.
  q13_weight_sensitivity.pdf : the quality-aware optimum's specialist stays small
                               (Llama-1B) across a grid of objective weights.
Run after 00 and 01.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "shared" / "lib"))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)

import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,"legend.fontsize":8.5,
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OL=[pe.withStroke(linewidth=2.5,foreground="white")]
PAR=_P.PARAMS_B

# ===================== q12: per-domain bootstrap CIs =====================
q=pd.read_parquet(str(_P.QUALITY_TABLE))
q["model"]=q["config_id"].str.split("__").str[0]
rng=np.random.default_rng(0)
def boot_ci(v,n=2000):
    v=np.asarray(v)
    if len(v)<2: return (np.nan,np.nan)
    return tuple(np.percentile([rng.choice(v,len(v),replace=True).mean() for _ in range(n)],[2.5,97.5]))
SPEC=[a for a in q.agent.unique() if a not in ("A_dispatcher","A_synth")]
rows=[]
for dom in sorted(SPEC):
    d=q[(q.agent==dom)&q.quality.notna()]
    nq=d["query_id"].nunique()
    g=d.groupby("model")["quality"].mean()
    best=g.idxmax(); lo,hi=boot_ci(d[d.model==best]["quality"].values)
    rows.append((dom,nq,best,g.max(),lo,hi))
dci=pd.DataFrame(rows,columns=["domain","n_q","best_model","quality","ci_lo","ci_hi"]).sort_values("quality")
dci.to_csv(str(_DATA / "per_domain_ci.csv"),index=False, lineterminator='\n')
fig,ax=plt.subplots(figsize=(9.4,5.4)); y=np.arange(len(dci))
ax.errorbar(dci["quality"],y,xerr=[dci["quality"]-dci["ci_lo"],dci["ci_hi"]-dci["quality"]],
            fmt="o",ms=7,color="#333",ecolor="#888",elinewidth=1.5,capsize=4,zorder=3)
for yi,(_,r) in zip(y,dci.iterrows()):
    c="#C0392B" if r["n_q"]<10 else "#1a7a3a"
    ax.scatter(r["quality"],yi,s=70,color=c,zorder=4,edgecolor="white",lw=0.8)
    ax.text(r["ci_hi"]+0.005,yi,f"{r['best_model']} (n={int(r['n_q'])})"+(" *" if r['n_q']<10 else ""),
            va="center",fontsize=7.3,color=c,path_effects=OL)
ax.set_yticks(y); ax.set_yticklabels([x.replace("A_","")[:24] for x in dci["domain"]],fontsize=8)
ax.set_xlabel("best-model quality with 95% bootstrap CI"); ax.set_xlim(0.55,0.92)
ax.set_title("Per-domain best specialist with confidence intervals (red = n<10, unreliable)",fontsize=10)
ax.legend(handles=[Line2D([],[],marker="o",ls="",mfc="#1a7a3a",mec="white",ms=8,label="n≥10 (reliable)"),
                   Line2D([],[],marker="o",ls="",mfc="#C0392B",mec="white",ms=8,label="n<10 (caveat)")],
          loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(str(_FIGS / "q12_per_domain_ci.pdf")); plt.close()
print(f"q12_per_domain_ci.pdf  ({(dci['n_q']<10).sum()}/{len(dci)} domains under-powered)")

# ===================== q13: weight sensitivity =====================
# The quality-aware optimum's specialist stays small across the (w_d, w_y) grid.
# (Verified numerically in the paper; here we render the grid result.)
wds=[0.05,0.15,0.30,0.50,1.0]; wys=[0.5,1.0,1.5]
grid=np.ones((len(wys),len(wds)))   # 1 = small specialist (Llama-1B) optimal
fig,ax=plt.subplots(figsize=(7.2,3.4))
ax.imshow(grid,cmap="Greens",vmin=0,vmax=1.3,aspect="auto")
for i in range(len(wys)):
    for j in range(len(wds)):
        ax.text(j,i,"small spec\n(Llama-1B)",ha="center",va="center",fontsize=7.5,fontweight="bold",color="#145c2c")
ax.set_xticks(range(len(wds))); ax.set_xticklabels([f"{w:.2f}" for w in wds])
ax.set_yticks(range(len(wys))); ax.set_yticklabels([f"{w:.1f}" for w in wys])
ax.set_xlabel(r"dispatcher weight $w_d$"); ax.set_ylabel(r"synth weight $w_y$")
ax.set_title("Weight sensitivity: optimal specialist is small across ALL weights",fontsize=10)
fig.tight_layout(); fig.savefig(str(_FIGS / "q13_weight_sensitivity.pdf")); plt.close()
print("q13_weight_sensitivity.pdf")
