import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
import pandas as pd, numpy as np
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
fr=pd.read_csv(str(_P.PERF_FRONTIER))

SHORT={"smollm2-360m":"SmolLM2-360M","qwen2.5-0_5b":"Qwen2.5-0.5B","llama3.2-1b":"Llama-3.2-1B",
       "qwen2.5-1_5b":"Qwen2.5-1.5B","smollm2-1_7b":"SmolLM2-1.7B","gemma2-2b":"Gemma-2-2B",
       "qwen2.5-3b":"Qwen2.5-3B","llama3.2-3b":"Llama-3.2-3B","ministral-3b":"Granite-3B",
       "phi3.5-mini":"Phi-3.5-mini","mistral-7b":"Mistral-7B"}
def sm(cfg): return SHORT[cfg.split("__")[0]]

# ============ FIG 7 (style): annotated frontier, k=3 highlighted, all k faint ============
fig,ax=plt.subplots(figsize=(8.8,5.0))
kcol={1:"#9CC0E6",3:"#C0392B",5:"#E0A24A",9:"#7FB39B"}
# faint curves for k=1,5,9; bold annotated for k=3
for k in [1,5,9]:
    d=fr[fr["k"]==k].sort_values("eps"); cap=d["capacity"].cummax()
    ax.step(d["eps"],cap,where="post",color=kcol[k],lw=1.3,alpha=0.55,zorder=2,label=f"k={k}")
k=3
d=fr[fr["k"]==k].sort_values("eps").reset_index(drop=True); cap=d["capacity"].cummax()
ax.step(d["eps"],cap,where="post",color=kcol[3],lw=2.2,zorder=4,label="k=3 (typical)")
ax.fill_between(d["eps"],cap,step="post",alpha=0.07,color=kcol[3],zorder=1)
# knots + annotate specialist-tier changes for k=3
prev_cap=None; prev_spec=None
for _,r in d.iterrows():
    if r["capacity"]!=prev_cap:
        ax.scatter([r["eps"]],[r["capacity"]],s=42,c=kcol[3],edgecolor="white",lw=0.8,zorder=5)
        spec=r["spec"].split("__")[0]
        if spec!=prev_spec:
            ax.annotate(f"spec → {SHORT[spec]}",(r["eps"],r["capacity"]),fontsize=7.4,ha="left",va="top",
                        xytext=(5,-4),textcoords="offset points",color="#7a1f15",path_effects=OL,zorder=6)
            prev_spec=spec
        prev_cap=r["capacity"]
ax.axhline(12.2,ls=":",color="#888",lw=1); ax.text(34.5,12.4,"memory-bound ceiling ≈ 12.2 B",fontsize=7.5,ha="right",color="#666")
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)  =  $L_{\mathrm{disp}}(15\,\mathrm{tok}) + k\,L_{\mathrm{spec}}(384) + L_{\mathrm{synth}}(384)$",fontsize=9.5)
ax.set_ylabel("max pipeline capacity (Σ params of 3 distinct slot-models, B)")
ax.set_xlim(2,35); ax.set_ylim(0,13.5)
ax.legend(title="activated specialists", frameon=True, loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig(f"{OUT}/frontier_annotated.pdf"); plt.close()

# ============ FIG 8 (style): critical-path latency breakdown, k=3 ============
fig,ax=plt.subplots(figsize=(8.8,4.8))
d=fr[fr["k"]==3].sort_values("eps").reset_index(drop=True)
# knots only (distinct capacity)
kn=[]; prev=None
for _,r in d.iterrows():
    if r["capacity"]!=prev: kn.append(r); prev=r["capacity"]
kn=pd.DataFrame(kn)
eps=kn["eps"].values; Ld=kn["Ld"].values; Ls3=3*kn["Ls"].values; Ly=kn["Ly"].values
w=0.6
ax.bar(eps,Ld,width=w,label=r"$L_{\mathrm{dispatcher}}$ (router, 15 tok)",color="#2E6FB0",edgecolor="white",lw=0.4,zorder=3)
ax.bar(eps,Ls3,width=w,bottom=Ld,label=r"$k\cdot L_{\mathrm{specialist}}$ ($k{=}3$, 384 tok each)",color="#E89A2C",edgecolor="white",lw=0.4,zorder=3)
ax.bar(eps,Ly,width=w,bottom=Ld+Ls3,label=r"$L_{\mathrm{synthesiser}}$ (384 tok)",color="#5B9C53",edgecolor="white",lw=0.4,zorder=3)
ax.plot(eps,eps,ls=(0,(4,3)),color="#444",lw=1.1,label=r"budget $\varepsilon$ (binds when tight)",zorder=4)
ax.set_xlabel(r"system latency budget $\varepsilon$ (s)"); ax.set_ylabel("critical-path latency (s)")
ax.legend(loc="upper left",frameon=True)
ax.set_xlim(5,21)
fig.tight_layout(); fig.savefig(f"{OUT}/chain_breakdown.pdf"); plt.close()
print("written: frontier_annotated.pdf, chain_breakdown.pdf")

# ============ knot table (k=3) for LaTeX ============
print("\n=== LaTeX table rows (k=3) ===")
prev=None
for _,r in d.iterrows():
    if r["capacity"]!=prev:
        sp=r["spec"].split("__"); q=sp[1].replace("_K_M","").replace("_0","")
        print(f"{r['eps']:.1f} & {r['capacity']:.1f} & {r['used_mem']:.2f} & {sm(r['disp'])} & {SHORT[sp[0]]} / {q} & {sm(r['synth'])} \\\\")
        prev=r["capacity"]
