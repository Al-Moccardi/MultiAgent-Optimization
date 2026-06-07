import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,"legend.fontsize":8.5,
    "xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,"grid.linewidth":0.5,
    "axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OUT=str(_FIGS)
r=pd.DataFrame(json.load(open(str(_DATA / "baseline_v4.json"))))

fig,axes=plt.subplots(1,2,figsize=(9.6,4.4))
# LEFT: MILP vs greedy quality curves for k=3 and k=9
ax=axes[0]
for k,c,ls in [(3,"#C0392B","-"),(9,"#1D6FB0","-")]:
    d=r[r["k"]==k].sort_values("eps")
    ax.plot(d["eps"],d["milp_q"],color=c,lw=2.0,ls=ls,label=f"MILP (k={k})",zorder=3)
    ax.plot(d["eps"],d["greedy_q"],color=c,lw=1.4,ls=(0,(3,2)),alpha=0.8,label=f"greedy (k={k})",zorder=2)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)"); ax.set_ylabel("pipeline quality (sum of per-domain affinities)")
ax.set_xlim(5,55); ax.legend(frameon=True,loc="lower right",fontsize=8)
ax.set_title("MILP vs greedy under heterogeneous specialists",fontsize=10)

# RIGHT: win-rate vs k (bar)
ax=axes[1]
from collections import defaultdict
byk=defaultdict(lambda:[0,0])
for _,x in r.iterrows():
    byk[x["k"]][1]+=1
    if pd.isna(x["greedy_q"]) or x["milp_q"]>x["greedy_q"]+1e-6: byk[x["k"]][0]+=1
ks=sorted(byk); wr=[100*byk[k][0]/byk[k][1] for k in ks]
bars=ax.bar([str(k) for k in ks],wr,color=["#9CC0E6","#6BA3D6","#C0392B","#8E2418"],edgecolor="white",lw=0.6,zorder=3)
for b,w in zip(bars,wr): ax.text(b.get_x()+b.get_width()/2,w+1.5,f"{w:.0f}%",ha="center",fontsize=9,fontweight="bold")
ax.set_xlabel("activated specialists k"); ax.set_ylabel("% points where MILP beats greedy")
ax.set_ylim(0,108); ax.set_title("Optimization advantage grows with k",fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/baseline_hetero.pdf"); plt.close()
print("written baseline_hetero.pdf")
