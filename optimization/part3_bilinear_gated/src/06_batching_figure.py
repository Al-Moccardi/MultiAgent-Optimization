"""
part3 / 06_batching_figure.py
=============================
Draws q18_batching_robustness.pdf from results/data/sigma_sweep.csv (produced by
05_batching_robustness.py). v1 of this package shipped the figure without its
plotting code; this script closes that gap so every paper figure regenerates
from the released data. Run after 05.
"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "shared" / "lib"))
import paths as _P
_FIGS, _DATA = _P.part_dirs(__file__)

import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10.5,
    "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.18, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150, "axes.axisbelow": True})

df = pd.read_csv(str(_DATA / "sigma_sweep.csv"))
k = 3
fig, ax = plt.subplots(figsize=(9.2, 5.2))
style = {1.0: ("#C0392B", "-", 2.4, r"$\sigma$ = 1 (sequential, worst case)"),
         0.66: ("#E89A2C", "-", 1.8, r"$\sigma$ = 0.66 (partial batch)"),
         0.5: ("#2E6FB0", "-", 1.8, r"$\sigma$ = 0.5 (partial batch)"),
         1.0 / 3.0: ("#1a7a3a", "-", 2.0, r"$\sigma$ = 1/k (ideal batch, best case)")}
for sig in sorted(df["sigma"].unique(), reverse=True):
    key = min(style, key=lambda s: abs(s - sig))
    c, ls, lw, lab = style[key]
    d = df[df["sigma"] == sig].sort_values("eps")
    ax.step(d["eps"], d["Q"].cummax(), where="post", color=c, ls=ls, lw=lw, label=lab, zorder=3)
# mark where the optimal specialist is NOT small (large-specialist picks)
PAR = _P.PARAMS_B
big = df[df["spec"].map(PAR) > 1.2]
if len(big):
    ax.scatter(big["eps"], big["Q"], s=60, facecolor="none", edgecolor="#111", lw=1.4,
               zorder=5, label="large specialist chosen")
ax.set_xlabel(r"system latency budget $\varepsilon$ (s, $k{=}3$)")
ax.set_ylabel("measured pipeline quality")
ax.set_title("Batching robustness: the frontier shifts left but the optimum is unchanged",
             fontsize=10.5)
ax.legend(loc="lower right", frameon=True)
fig.tight_layout()
fig.savefig(str(_FIGS / "q18_batching_robustness.pdf"))
plt.close()
print("written q18_batching_robustness.pdf")
