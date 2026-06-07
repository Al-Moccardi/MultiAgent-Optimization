import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,
    "legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,
    "grid.linewidth":0.5,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OL=[pe.withStroke(linewidth=2,foreground="white")]
OUT=str(_FIGS)

# ---------- FIG A: corrected frontier, 4 k-curves ----------
fr=pd.read_csv(str(_P.PERF_FRONTIER))
fig,ax=plt.subplots(figsize=(8.8,5.0))
kcol={1:"#2E6FB0",3:"#1D9E75",5:"#E89A2C",9:"#C0392B"}
for k in [1,3,5,9]:
    d=fr[fr["k"]==k].sort_values("eps")
    # staircase of best capacity vs eps (cap is nondecreasing in eps already)
    cummax=d["capacity"].cummax()
    ax.step(d["eps"],cummax,where="post",color=kcol[k],lw=1.9,label=f"k = {k} activated",zorder=3)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)  =  $L_{\mathrm{disp}}(15\,\mathrm{tok}) + k\,L_{\mathrm{spec}}(384) + L_{\mathrm{synth}}(384)$",fontsize=9.5)
ax.set_ylabel("max pipeline capacity (Σ params of 3 distinct slot-models, B)")
ax.set_xlim(2,55); ax.set_ylim(0,13)
ax.axhline(12.2,ls=":",color="#888",lw=1); ax.text(53,12.3,"memory-bound ceiling ≈12.2 B",fontsize=7.5,ha="right",color="#666")
ax.legend(title="specialists on critical path",frameon=True,loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/syscap3_frontier.pdf"); plt.close()

# ---------- FIG B: MILP vs baseline ----------
cmp=json.load(open(str(_DATA / "baseline_cmp.json")))
d=pd.DataFrame(cmp)
fig,ax=plt.subplots(figsize=(8.8,4.8))
# show k=3 and k=9 as representative; plot milp, greedy, uniform
for k,mk in [(3,"o"),(9,"s")]:
    dk=d[d["k"]==k].sort_values("eps")
    ax.step(dk["eps"],dk["milp"].cummax(),where="post",color="#C0392B",lw=1.8,
            label=f"MILP (k={k})" if k==3 else None,zorder=4,alpha=1 if k==3 else 0.5,
            ls="-" if k==3 else "--")
    ax.step(dk["eps"],dk["greedy"].cummax(),where="post",color="#2E6FB0",lw=1.4,
            label=f"greedy heuristic (k={k})" if k==3 else None,zorder=3,alpha=1 if k==3 else 0.5,
            ls="-" if k==3 else "--")
    ax.step(dk["eps"],dk["uniform"].fillna(0).cummax(),where="post",color="#999",lw=1.2,
            label=f"uniform (1 model)" if k==3 else None,zorder=2,alpha=1 if k==3 else 0.5,
            ls="-" if k==3 else "--")
# mark the MILP-wins
wins=[r for r in cmp if r["greedy"] and r["milp"]>r["greedy"]+1e-6]
ax.scatter([r["eps"] for r in wins],[r["milp"] for r in wins],s=55,facecolor="none",
           edgecolor="#C0392B",lw=1.5,zorder=6,label="MILP > greedy (7 pts)")
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)"); ax.set_ylabel("capacity (B)")
ax.set_xlim(2,55); ax.set_ylim(0,13)
ax.legend(frameon=True,loc="lower right",ncol=1,fontsize=8)
ax.set_title("MILP vs heuristics: near-identical except a narrow mid-latency band",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_cmp.pdf"); plt.close()

# ---------- FIG C: uncertainty band ----------
un=json.load(open(str(_DATA / "uncertainty_k3.json")))
u=pd.DataFrame(un)
fig,ax=plt.subplots(figsize=(8.0,4.6))
ax.fill_between(u["eps"],u["p10"],u["p90"],alpha=0.22,color="#1D9E75",label="10–90% band",step="post")
ax.step(u["eps"],u["p50"],where="post",color="#0F6E56",lw=1.8,label="median",zorder=3)
ax.step(u["eps"],u["base"],where="post",color="#C0392B",lw=1.3,ls="--",label="point-estimate frontier",zorder=4)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)"); ax.set_ylabel("max capacity (B), k=3")
ax.set_title("Frontier stability under ±10% throughput/TTFT noise (25 draws)",fontsize=10)
ax.legend(frameon=True,loc="lower right"); ax.set_ylim(7,13)
fig.tight_layout(); fig.savefig(f"{OUT}/uncertainty.pdf"); plt.close()
print("3 figures written: syscap3_frontier, baseline_cmp, uncertainty")
