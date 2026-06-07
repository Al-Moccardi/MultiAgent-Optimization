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
SHORT={"smollm2-360m":"SmolLM2-0.36B","qwen2.5-0_5b":"Qwen2.5-0.5B","llama3.2-1b":"Llama3.2-1B",
 "qwen2.5-1_5b":"Qwen2.5-1.5B","smollm2-1_7b":"SmolLM2-1.7B","gemma2-2b":"Gemma2-2B","qwen2.5-3b":"Qwen2.5-3B",
 "llama3.2-3b":"Llama3.2-3B","ministral-3b":"Granite-3B(MoE)","phi3.5-mini":"Phi3.5-3.8B","mistral-7b":"Mistral-7B"}

# ===== PLOT 5: dispatcher routing F1 vs params (opposite trend to specialists) =====
g=pd.read_parquet(str(_DATA / "dispatcher_f1.parquet")).reset_index()
fig,ax=plt.subplots(figsize=(8.8,5.0))
cmap=plt.cm.plasma(np.linspace(0.1,0.9,len(g)))
bars=ax.bar(range(len(g)),g["f1"],color=cmap,edgecolor="white",lw=0.6,zorder=3)
for i,(_,r) in enumerate(g.iterrows()):
    ax.text(i,r["f1"]+0.008,f"{r['f1']:.2f}",ha="center",fontsize=8,fontweight="bold")
    ax.text(i,0.01,f"{r['params']:.2g}B",ha="center",va="bottom",fontsize=6.8,color="white",rotation=90,fontweight="bold")
ax.set_xticks(range(len(g))); ax.set_xticklabels([SHORT[m] for m in g["model"]],rotation=32,ha="right",fontsize=8)
ax.set_ylabel("routing F1 (dispatcher)"); ax.set_ylim(0,0.78)
ax.set_title("Dispatcher: routing F1 rewards LARGER models (opposite of specialists)",fontsize=10.5)
ax.tick_params(axis="x",length=0)
fig.tight_layout(); fig.savefig(str(_FIGS / "q5_dispatcher_f1.pdf")); plt.close()

# ===== PLOT 6: per-agent best/worst model heatmap-style (quality leaders differ) =====
m=pd.read_parquet(str(_DATA / "quality_cost_merged.parquet"))
gen=m[(m["agent"]!="A_dispatcher")&(m["agent"]!="A_synth")].copy()
gen["dom"]=gen["agent"].str.replace("A_","",regex=False)
piv=gen.groupby(["dom","model"])["quality"].mean().unstack("model")
MODELS=sorted([c for c in piv.columns],key=lambda mm:{ "smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,"gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}[mm])
piv=piv[MODELS]
fig,ax=plt.subplots(figsize=(10.0,5.0))
im=ax.imshow(piv.values,aspect="auto",cmap="RdYlGn",vmin=0.45,vmax=0.80)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([SHORT[m] for m in MODELS],rotation=35,ha="right",fontsize=8)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([d[:24] for d in piv.index],fontsize=8)
# mark best per row
for i in range(len(piv.index)):
    row=piv.values[i]; 
    if np.all(np.isnan(row)): continue
    j=np.nanargmax(row)
    ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="#111",lw=2.2))
    for jj in range(len(MODELS)):
        if not np.isnan(row[jj]): ax.text(jj,i,f"{row[jj]:.2f}",ha="center",va="center",fontsize=6.3,color="#222")
cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02); cb.set_label("quality",fontsize=8)
ax.set_title("Per-domain specialist quality (black box = best model for that domain)",fontsize=10.5)
fig.tight_layout(); fig.savefig(str(_FIGS / "q6_agent_matrix.pdf")); plt.close()
print("written q5_dispatcher_f1.pdf, q6_agent_matrix.pdf")
