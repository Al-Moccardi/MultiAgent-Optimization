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
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,"gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
q=pd.read_parquet(str(_P.QUALITY_TABLE))
q["model"]=q["config_id"].str.split("__").str[0]; q["params"]=q["model"].map(PAR)
MODELS=sorted(PAR,key=PAR.get)

# ===== PLOT 7: THREE REGIMES — the key new figure =====
fig,ax=plt.subplots(figsize=(9.2,5.4))
def role_means(role, exclude=False):
    if exclude: d=q[(~q["agent"].isin(["A_dispatcher","A_synth"]))&q["quality"].notna()]
    else: d=q[(q["agent"]==role)&q["quality"].notna()]
    g=d.groupby("model").agg(p=("params","first"),q=("quality","mean")).sort_values("p")
    return g
disp=role_means("A_dispatcher"); syn=role_means("A_synth"); spec=role_means(None,exclude=True)
ax.plot(disp["p"],disp["q"],"-o",color="#C0392B",lw=2,ms=6,label="Dispatcher (routing F1)",zorder=4)
ax.plot(syn["p"],syn["q"],"-s",color="#2E6FB0",lw=2,ms=6,label="Synthesiser (RAG quality)",zorder=4)
ax.plot(spec["p"],spec["q"],"-^",color="#1a7a3a",lw=2,ms=6,label="Specialists (RAG quality)",zorder=4)
# annotate the divergence
ax.annotate("specialists peak\nat 1.2B then plateau",(1.2,spec.loc["llama3.2-1b","q"]),fontsize=8,color="#145c2c",
            fontweight="bold",xytext=(10,-26),textcoords="offset points",path_effects=OL)
ax.annotate("dispatcher & synth\nkeep rising with size",(6.0,0.70),fontsize=8,color="#1a3f6b",
            fontweight="bold",ha="center",xytext=(0,-44),textcoords="offset points",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#1a3f6b",lw=1))
ax.margins(x=0.10,y=0.08)  # headroom so the right-hand label clears the frame
ax.set_xlabel("model parameters (B)"); ax.set_ylabel("measured quality / routing F1")
ax.set_title("Three roles, three scaling laws: only specialists prefer small models",fontsize=10.5)
ax.legend(loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(str(_FIGS / "q7_three_regimes.pdf")); plt.close()

# ===== PLOT 8: synthesiser quality vs params (was TODO) =====
fig,ax=plt.subplots(figsize=(8.6,4.8))
sy=q[(q["agent"]=="A_synth")&q["quality"].notna()]
cmap=dict(zip(MODELS,plt.cm.cividis(np.linspace(0.05,0.95,len(MODELS)))))
for mdl in MODELS:
    d=sy[sy["model"]==mdl]
    if len(d): ax.scatter(d["params"],d["quality"],s=55,color=cmap[mdl],alpha=0.6,edgecolor="white",lw=0.4,zorder=3)
ax.plot(syn["p"],syn["q"],"-o",color="#222",lw=1.5,ms=5,zorder=4,label="mean per model")
z=np.polyfit(sy["params"],sy["quality"],1); xs=np.linspace(0.3,7.2,40)
ax.plot(xs,np.polyval(z,xs),"--",color="#2E6FB0",lw=1.5,label=f"fit (r={sy['params'].corr(sy['quality']):.2f})")
ax.annotate(f"Mistral-7B best\n(q={syn.loc['mistral-7b','q']:.2f})",(7.2,syn.loc["mistral-7b","q"]),fontsize=8,
            fontweight="bold",color="#1a3f6b",xytext=(-30,-28),textcoords="offset points",path_effects=OL)
ax.set_xlabel("model parameters (B)"); ax.set_ylabel("synthesiser quality")
ax.set_title("Synthesiser quality rises with size (largest model best)",fontsize=10.5)
ax.legend(loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(str(_FIGS / "q8_synth_quality.pdf")); plt.close()
print("written q7_three_regimes.pdf, q8_synth_quality.pdf")
