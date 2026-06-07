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
PAR={"smollm2-360m":0.36,"qwen2.5-0_5b":0.5,"llama3.2-1b":1.2,"qwen2.5-1_5b":1.5,"smollm2-1_7b":1.7,"gemma2-2b":2.6,"qwen2.5-3b":3.0,"llama3.2-3b":3.2,"ministral-3b":3.3,"phi3.5-mini":3.8,"mistral-7b":7.2}
SHORT={"smollm2-360m":"SmolLM2-0.36B","qwen2.5-0_5b":"Qwen2.5-0.5B","llama3.2-1b":"Llama3.2-1B","qwen2.5-1_5b":"Qwen2.5-1.5B","smollm2-1_7b":"SmolLM2-1.7B","gemma2-2b":"Gemma2-2B","qwen2.5-3b":"Qwen2.5-3B","llama3.2-3b":"Llama3.2-3B","ministral-3b":"Granite-3B(MoE)","phi3.5-mini":"Phi3.5-3.8B","mistral-7b":"Mistral-7B"}

# ===== PLOT A: synthesiser RAGAS metrics decomposed =====
sc=pd.read_csv(str(_P.SCORECARD))
syn=sc[sc["agent"]=="A_synth"].copy(); syn["model"]=syn["config_id"].str.split("__").str[0]; syn["params"]=syn["model"].map(PAR)
g=syn.groupby("model").agg(params=("params","first"),faith=("faithfulness","mean"),rel=("answer_relevancy","mean"),
    corr=("correctness","mean"),q=("quality","mean")).sort_values("params")
fig,ax=plt.subplots(figsize=(9.4,5.2))
ax.plot(g["params"],g["corr"],"-o",color="#1a7a3a",lw=2,ms=6,label="correctness (drives quality)",zorder=5)
ax.plot(g["params"],g["rel"],"-s",color="#2E6FB0",lw=2,ms=5,label="answer relevancy",zorder=4)
ax.plot(g["params"],g["faith"],"-^",color="#C0392B",lw=2,ms=5,label="faithfulness (DROPS with size)",zorder=4)
ax.axhline(0.816,ls=":",color="#888",lw=1.2); ax.text(7.1,0.826,"context precision = 0.82 (retriever-fixed)",fontsize=7,ha="right",color="#666")
ax.axhline(0.518,ls=":",color="#aaa",lw=1.2); ax.text(7.1,0.528,"context recall = 0.52 (retriever-fixed)",fontsize=7,ha="right",color="#888")
ax.annotate("large models: less faithful\nbut more correct\n(use parametric knowledge)",(7.2,g.loc["mistral-7b","faith"]),
            fontsize=7.5,color="#922",fontweight="bold",xytext=(-120,30),textcoords="offset points",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#922",lw=1))
ax.set_xlabel("synthesiser model parameters (B)"); ax.set_ylabel("RAGAS metric value")
ax.set_title("Synthesiser quality decomposed: correctness rises, faithfulness falls",fontsize=10.5)
ax.legend(loc="lower right",frameon=True,framealpha=0.95); ax.set_ylim(0.45,0.90)
fig.tight_layout(); fig.savefig(f"{OUT}/q10_synth_ragas.pdf"); plt.close()

# ===== PLOT B: proxy vs quality-optimal (pipeline-level, frontal comparison) =====
# Self-contained: build the pipeline-level comparison (dispatcher+k*spec+synth) at
# matched latency, choosing each role by max-params (proxy) vs max-quality.
def _build_proxy_vs_quality():
    sc2=pd.read_csv(str(_P.SCORECARD)); pf=pd.read_parquet(str(_P.PERF_TABLE))
    pf["model"]=pf["config_id"].str.split("__").str[0]
    pf["lg"]=pf["ttft_s"]+_P.TOK_GENERATION/pf["throughput_tok_s"]
    pf["ld"]=pf["ttft_s"]+_P.TOK_ROUTING/pf["throughput_tok_s"]
    pf=pf[pf["peak_mem_gb"]<=_P.MEM_BUDGET_GB]
    lg={r.config_id:r.lg for r in pf.itertuples()}; ld={r.config_id:r.ld for r in pf.itertuples()}
    mo={r.config_id:r.model for r in pf.itertuples()}; mem={r.config_id:r.peak_mem_gb for r in pf.itertuples()}
    feas=set(pf["config_id"])
    SP=[a for a in sc2.agent.unique() if a not in ("A_dispatcher","A_synth")]
    F=dict((r.config_id,r.quality) for r in sc2[sc2.agent=="A_dispatcher"].itertuples() if r.config_id in feas)
    Y=dict((r.config_id,r.quality) for r in sc2[sc2.agent=="A_synth"].itertuples() if r.config_id in feas)
    Sm={}
    for r in sc2[sc2.agent.isin(SP)].itertuples():
        if r.config_id in feas: Sm.setdefault(r.config_id,[]).append(r.quality)
    Sm={c:np.mean(v) for c,v in Sm.items()}
    rows=[]
    for eps in np.arange(7.0,20.01,0.5):
        eps=round(eps,1); best={"param":(-1,None),"qual":(-1,None)}
        for fy in [0.3,0.4,0.5,0.6]:
            es=(eps-0.5)/4
            for metric,Fsel,Ssel,Ysel in [("param",PAR,PAR,PAR),("qual",F,Sm,Y)]:
                cd=[c for c in F if ld[c]<=1.0]; cs=[c for c in Sm if lg[c]<=es]; cy=[c for c in Y if lg[c]<=es]
                if not(cd and cs and cy): continue
                if metric=="param":
                    d_=max(cd,key=lambda c:PAR[mo[c]]); s_=max(cs,key=lambda c:PAR[mo[c]]); y_=max(cy,key=lambda c:PAR[mo[c]])
                else:
                    d_=max(cd,key=lambda c:F[c]); s_=max(cs,key=lambda c:Sm[c]); y_=max(cy,key=lambda c:Y[c])
                if mem[d_]+(0 if mo[s_]==mo[d_] else mem[s_])+(0 if mo[y_] in(mo[d_],mo[s_]) else mem[y_])>_P.MEM_BUDGET_GB: continue
                if ld[d_]+3*lg[s_]+lg[y_]>eps: continue
                Q=0.15*F[d_]+Sm[s_]+Y[y_]
                key="param" if metric=="param" else "qual"
                if Q>best[key][0]: best[key]=(Q,(mo[d_],mo[s_],mo[y_]))
        if best["param"][1] and best["qual"][1]:
            rows.append(dict(eps=eps,Q_proxy=best["param"][0],Q_quality=best["qual"][0]))
    return pd.DataFrame(rows)
d=_build_proxy_vs_quality()
d.to_csv(str(_DATA / "proxy_vs_quality_pipeline.csv"),index=False, lineterminator='\n')
fig,ax=plt.subplots(figsize=(9.4,5.4))
ax.step(d["eps"],d["Q_quality"].cummax(),where="post",color="#1a7a3a",lw=2.4,label="quality-optimal MILP",zorder=5)
ax.step(d["eps"],d["Q_proxy"],where="post",color="#C0392B",lw=2.0,label="parameter-proxy allocation",zorder=4)
ax.fill_between(d["eps"],d["Q_proxy"],d["Q_quality"].cummax(),step="post",alpha=0.15,color="#1a7a3a",zorder=2)
_mid=d["eps"].iloc[len(d)//2]
ax.annotate("proxy picks a huge dispatcher,\nstarves specialists+synth\n(quality drops as budget grows)",(_mid,d[d.eps==_mid]["Q_proxy"].iloc[0]),
            fontsize=7.5,color="#922",fontweight="bold",xytext=(20,40),textcoords="offset points",path_effects=OL,
            arrowprops=dict(arrowstyle="->",color="#922",lw=1))
ax.set_xlabel(r"system latency budget $\varepsilon$ (s, $k{=}3$)"); ax.set_ylabel("measured pipeline quality")
_gap=d["Q_quality"].cummax()-d["Q_proxy"]
ax.set_title(f"Parameter proxy vs measured-quality optimum (mean gap {_gap.mean():.2f}, max {_gap.max():.2f})",fontsize=10.5); ax.legend(loc="upper left",frameon=True,framealpha=0.95)
ax.legend(loc="lower right",frameon=True)
fig.tight_layout(); fig.savefig(f"{OUT}/q11_proxy_vs_quality.pdf"); plt.close()
print("written q10_synth_ragas.pdf, q11_proxy_vs_quality.pdf")
