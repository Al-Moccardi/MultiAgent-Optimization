import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / 'shared' / 'lib'))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,
    "legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,
    "grid.linewidth":0.5,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})

p=pd.read_parquet(str(_P.PERF_TABLE))
p["model"]=p["config_id"].str.split("__").str[0]; p["quant"]=p["config_id"].str.split("__").str[1]
p["ctx"]=p["config_id"].str.split("__c").str[1].astype(int)
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
p["params"]=p["model"].map(PAR); T=384
p["lat384"]=p["ttft_s"]+T/p["throughput_tok_s"]; p["e384"]=p["energy_j_per_tok"]*T
FAM={'smollm2-360m':'SmolLM2','smollm2-1_7b':'SmolLM2','qwen2.5-0_5b':'Qwen2.5','qwen2.5-1_5b':'Qwen2.5',
     'qwen2.5-3b':'Qwen2.5','llama3.2-1b':'Llama-3.2','llama3.2-3b':'Llama-3.2','gemma2-2b':'Gemma-2',
     'phi3.5-mini':'Phi-3.5','ministral-3b':'Granite(MoE)','mistral-7b':'Mistral'}
p["family"]=p["model"].map(FAM)
fam_order=sorted(p["family"].unique(),key=lambda f:p[p["family"]==f]["params"].min())
FAMC=dict(zip(fam_order,plt.cm.tab10(np.linspace(0,1,len(fam_order)))))
ref=p[(p["quant"]=="Q5_K_M")&(p["ctx"]==4096)].copy()
OUT=str(_FIGS)

# wider bars + larger gap; size label placed VERTICALLY inside each bar (no collision);
# family name on its own row well below the axis.
fig,axes=plt.subplots(1,2,figsize=(10.0,5.0))
BARW=0.82; GAP=2.0
for ax,col,ylab,fmt,toppad in [
    (axes[0],"lat384","latency for a 384-token answer (s)","{:.1f}",0.30),
    (axes[1],"e384","energy for a 384-token answer (J)","{:.0f}",9.0)]:
    xi=0; fam_centers=[]; fam_names=[]
    ymax=ref[col].max()
    for f in fam_order:
        g=ref[ref["family"]==f].sort_values("params")
        if len(g)==0: continue
        xs=np.arange(len(g))*1.02+xi
        ax.bar(xs,g[col],width=BARW,color=FAMC[f],edgecolor="white",lw=0.6,zorder=3)
        for x,(_,r) in zip(xs,g.iterrows()):
            ax.text(x,r[col]+toppad,fmt.format(r[col]),ha="center",va="bottom",fontsize=7.6,zorder=4)
            # size label vertical inside the bar; if the bar is too short to hold it,
            # place it just above the bar base in the bar's colour instead (no clipping)
            if r[col] >= ymax*0.16:
                ax.text(x,ymax*0.03,f"{r['params']:.2g}B",ha="center",va="bottom",fontsize=7,
                        color="white",rotation=90,zorder=5,fontweight="bold")
            else:
                ax.text(x,r[col]+toppad,"",ha="center")  # value already drawn above
                ax.text(x,ymax*0.005,f"{r['params']:.2g}B",ha="center",va="bottom",fontsize=6.5,
                        color=FAMC[f],rotation=90,zorder=5)
        fam_centers.append(xs.mean()); fam_names.append(f)
        xi=xs[-1]+GAP
    ax.set_ylabel(ylab)
    ax.set_ylim(0,ymax*1.16)
    ax.set_xticks(fam_centers)
    ax.set_xticklabels(fam_names,rotation=22,ha="right",fontsize=8.5)
    ax.tick_params(axis="x",length=0,pad=2)
axes[0].set_title("Latency",fontsize=11)
axes[1].set_title("Energy",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/family_cost.pdf"); plt.close()
print("family_cost.pdf rewritten (no overlap)")

# ===== CORRELATION HEATMAP =====
cols=["params","peak_mem_gb","ttft_s","throughput_tok_s","energy_j_per_tok","lat384"]
nice={"params":"params (B)","peak_mem_gb":"peak VRAM","ttft_s":"TTFT",
      "throughput_tok_s":"throughput","energy_j_per_tok":"energy/token","lat384":"latency@384"}
C=p[cols].corr()  # Pearson
labels=[nice[c] for c in cols]
fig,ax=plt.subplots(figsize=(6.6,5.4))
im=ax.imshow(C.values,cmap="RdBu_r",vmin=-1,vmax=1,aspect="equal")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels,rotation=35,ha="right",fontsize=9)
ax.set_yticks(range(len(cols))); ax.set_yticklabels(labels,fontsize=9)
for i in range(len(cols)):
    for j in range(len(cols)):
        v=C.values[i,j]
        ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=8.5,
                color="white" if abs(v)>0.55 else "black",
                fontweight="bold" if (i!=j and abs(v)>0.7) else "normal")
ax.set_xticks(np.arange(-.5,len(cols),1),minor=True); ax.set_yticks(np.arange(-.5,len(cols),1),minor=True)
ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Pearson correlation",fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/corr_heatmap.pdf"); plt.close()
print("corr_heatmap.pdf written")
