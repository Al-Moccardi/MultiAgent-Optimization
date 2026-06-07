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
import matplotlib.colors as mcolors

rcParams.update({"font.family":"serif","font.size":10,"axes.labelsize":10.5,
    "legend.fontsize":8.5,"xtick.labelsize":9,"ytick.labelsize":9,"axes.grid":True,"grid.alpha":0.18,
    "grid.linewidth":0.5,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150,"axes.axisbelow":True})
OL=[pe.withStroke(linewidth=1.8,foreground="white")]

p=pd.read_parquet(str(_P.PERF_TABLE))
p["model"]=p["config_id"].str.split("__").str[0]; p["quant"]=p["config_id"].str.split("__").str[1]
p["ctx"]=p["config_id"].str.split("__c").str[1].astype(int)
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,
     "gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
SHORT={"smollm2-360m":"SmolLM2-360M","qwen2.5-0_5b":"Qwen2.5-0.5B","llama3.2-1b":"Llama3.2-1B",
       "qwen2.5-1_5b":"Qwen2.5-1.5B","smollm2-1_7b":"SmolLM2-1.7B","gemma2-2b":"Gemma2-2B",
       "qwen2.5-3b":"Qwen2.5-3B","llama3.2-3b":"Llama3.2-3B","ministral-3b":"Granite(MoE)",
       "phi3.5-mini":"Phi3.5-mini","mistral-7b":"Mistral-7B"}
p["params"]=p["model"].map(PAR); T=384; M=6.99
p["lat384"]=p["ttft_s"]+T/p["throughput_tok_s"]; p["e384"]=p["energy_j_per_tok"]*T; p["tokJ"]=1/p["energy_j_per_tok"]
QC={"Q3_K_M":"#2E6FB0","Q5_K_M":"#E89A2C","Q8_0":"#C0392B"}; QM={"Q3_K_M":"o","Q5_K_M":"s","Q8_0":"D"}
models=sorted(p["model"].unique(),key=lambda m:PAR[m]); cmap=plt.cm.viridis
def csz(m): return cmap(np.log(PAR[m]/0.36)/np.log(7.2/0.36))
OUT=str(_FIGS); import os; os.makedirs(OUT,exist_ok=True)

# F1 heatmap (wide, no title)
fig,ax=plt.subplots(figsize=(9.2,4.6))
cols=[(q,c) for q in ["Q3_K_M","Q5_K_M","Q8_0"] for c in [2048,4096,8192]]
mat=np.full((len(models),len(cols)),np.nan)
for i,m in enumerate(models):
    for j,(q,c) in enumerate(cols):
        d=p[(p["model"]==m)&(p["quant"]==q)&(p["ctx"]==c)]
        if len(d):mat[i,j]=d["peak_mem_gb"].iloc[0]
cmap2=plt.cm.RdYlGn_r;cmap2.set_bad("#dddddd")
im=ax.imshow(mat,aspect="auto",cmap=cmap2,vmin=0,vmax=M*1.3)
for i in range(len(models)):
    for j in range(len(cols)):
        v=mat[i,j]
        if np.isnan(v):ax.text(j,i,"–",ha="center",va="center",fontsize=8,color="#888")
        else:
            over=v>M
            ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=7.6,
                color="white" if (v>M*0.82 or v<1.1) else "black",fontweight="bold" if over else "normal")
            if over:ax.add_patch(plt.Rectangle((j-0.5,i-0.5),1,1,fill=False,edgecolor="black",lw=2.2))
ax.set_xticks(range(len(cols)));ax.set_xticklabels([f"{q.replace('_K_M','').replace('_0','')}\n{c}" for q,c in cols],fontsize=8)
ax.set_yticks(range(len(models)));ax.set_yticklabels([SHORT[m] for m in models],fontsize=8.5)
for xv in [2.5,5.5]:ax.axvline(xv,color="white",lw=2.5)
cb=fig.colorbar(im,ax=ax,pad=0.012);cb.set_label("peak VRAM (GB)");cb.ax.axhline(M,color="black",lw=1.5)
fig.tight_layout();fig.savefig(f"{OUT}/heatmap.pdf");plt.close()

# F2 throughput power-law
fig,ax=plt.subplots(figsize=(7.2,4.4))
for q in QC:
    d=p[p["quant"]==q];ax.scatter(d["params"],d["throughput_tok_s"],s=42,c=QC[q],marker=QM[q],edgecolor="white",lw=0.5,alpha=0.9,zorder=3)
    med=d.groupby("params")["throughput_tok_s"].median();ax.plot(med.index,med.values,color=QC[q],lw=1.8,alpha=0.85,zorder=2)
ax.set_xscale("log");ax.set_yscale("log");ax.set_xticks([0.36,0.5,1,2,3,7]);ax.set_xticklabels(["0.36","0.5","1","2","3","7"])
ax.set_yticks([10,20,50,100,200]);ax.set_yticklabels(["10","20","50","100","200"])
ax.set_xlabel("model size (B parameters, log scale)");ax.set_ylabel("decode throughput (tok/s, log scale)")
fa=p.loc[p["throughput_tok_s"].idxmax()];sl=p.loc[p["throughput_tok_s"].idxmin()]
for r,ha in [(fa,"left"),(sl,"right")]:
    ax.annotate(f"{SHORT[r['model']]}\n{r['throughput_tok_s']:.0f} tok/s",(r["params"],r["throughput_tok_s"]),
        fontsize=8,ha=ha,va="center",xytext=(r["params"]*(1.06 if ha=='left' else .94),r["throughput_tok_s"]*0.62),
        textcoords="data",path_effects=OL,arrowprops=dict(arrowstyle="-",lw=0.6,color="#888"))
xx=np.array([0.36,7.2]);ax.plot(xx,150*(xx/0.5)**-0.7,ls=":",color="#999",lw=1.2)
ax.text(3.6,150*(3.6/0.5)**-0.7*1.25,r"$\sim N^{-0.7}$",fontsize=9.5,color="#777",rotation=-20)
ax.legend(handles=[Line2D([],[],marker=QM[q],color=QC[q],lw=1.8,ms=7,mec="white",label=q) for q in QC],
    title="quantization",frameon=True,framealpha=0.9,loc="upper right")
fig.tight_layout();fig.savefig(f"{OUT}/throughput.pdf");plt.close()

# F3 quant cost slopegraph (two panels)
fig,axes=plt.subplots(1,2,figsize=(8.6,4.2));order=["Q3_K_M","Q5_K_M","Q8_0"];xp=[0,1,2]
for m in models:
    d=p[(p["model"]==m)&(p["ctx"]==4096)].set_index("quant").reindex(order)
    if d["peak_mem_gb"].isna().all():continue
    c=csz(m)
    axes[0].plot(xp,d["peak_mem_gb"],"-o",color=c,ms=5,lw=1.6,alpha=0.9)
    axes[1].plot(xp,d["throughput_tok_s"],"-o",color=c,ms=5,lw=1.6,alpha=0.9)
    if not np.isnan(d["throughput_tok_s"].iloc[-1]):
        axes[1].text(2.05,d["throughput_tok_s"].iloc[-1],SHORT[m],fontsize=7,va="center",color=c,path_effects=OL)
axes[0].axhline(M,color="#444",ls=(0,(5,2)),lw=1.1);axes[0].text(0,M*1.02,"budget 6.99 GB",fontsize=8,color="#444")
for a in axes:a.set_xticks(xp);a.set_xticklabels(["Q3_K_M","Q5_K_M","Q8_0"]);a.set_xlim(-0.15,2.85)
axes[0].set_ylabel("peak VRAM (GB)");axes[0].set_xlabel("quantization")
axes[1].set_ylabel("throughput (tok/s)");axes[1].set_xlabel("quantization")
fig.tight_layout();fig.savefig(f"{OUT}/quant_cost.pdf");plt.close()

# F4 cost landscape 4D (wide)
fig,ax=plt.subplots(figsize=(9.0,5.2))
feas=p[p["peak_mem_gb"]<=M].copy().reset_index(drop=True)
sizes=40+(feas["peak_mem_gb"]/feas["peak_mem_gb"].max())*460
sc=ax.scatter(feas["lat384"],feas["e384"],s=sizes,c=feas["params"],cmap="plasma",
    norm=mcolors.LogNorm(vmin=0.36,vmax=7.2),edgecolor="white",lw=0.6,alpha=0.88,zorder=3)
def pmin(df,xc,yc):
    pts=df[[xc,yc]].values;k=np.ones(len(df),bool)
    for i in range(len(df)):
        if not k[i]:continue
        dom=(pts[:,0]<=pts[i,0])&(pts[:,1]<=pts[i,1])&((pts[:,0]<pts[i,0])|(pts[:,1]<pts[i,1]))
        if dom.any():k[i]=False
    return k
feas["par"]=pmin(feas,"lat384","e384");fr=feas[feas["par"]].sort_values("lat384")
ax.plot(fr["lat384"],fr["e384"],"-",color="#222",lw=1.4,zorder=4,alpha=0.7)
ax.scatter(fr["lat384"],fr["e384"],s=85,facecolor="none",edgecolor="#111",lw=1.5,zorder=5)
seen=set()
for _,r in fr.iterrows():
    if r["model"] in seen: continue
    seen.add(r["model"])
    ax.annotate(f"{SHORT[r['model']]}",(r["lat384"],r["e384"]),fontsize=7.5,ha="left",va="top",
        xytext=(5,-4),textcoords="offset points",path_effects=OL,zorder=6)
ax.set_xscale("log");ax.set_yscale("log")
ax.set_xlabel("latency for a 384-token answer (s, log scale)");ax.set_ylabel("energy for a 384-token answer (J, log scale)")
cb=fig.colorbar(sc,ax=ax,pad=0.012);cb.set_label("model size (B params)")
cb.set_ticks([0.36,0.5,1,2,3,7]);cb.set_ticklabels(["0.36","0.5","1","2","3","7"])
for mem,lab in [(1,"1 GB"),(3,"3 GB"),(6,"6 GB")]:
    ax.scatter([],[],s=40+(mem/p["peak_mem_gb"].max())*460,c="#bbb",edgecolor="white",label=lab)
l1=ax.legend(title="peak VRAM (bubble area)",loc="lower right",frameon=True,framealpha=0.92,labelspacing=1.3,borderpad=0.9)
ax.add_artist(l1)
ax.legend(handles=[Line2D([],[],marker="o",mfc="none",mec="#111",mew=1.5,ls="-",color="#222",label="latency–energy Pareto front")],loc="upper left",frameon=True)
fig.tight_layout();fig.savefig(f"{OUT}/cost_landscape.pdf");plt.close()

print("part1: performance-characterization figures written to results/figures")
