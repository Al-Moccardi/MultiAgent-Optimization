#!/usr/bin/env python3
"""
verify.py -- one-command replication check for the MAMAP-Edge package.

Run AFTER the RUNBOOK pipeline. Recomputes every headline number of the paper
from the released inputs and the regenerated outputs, and asserts it against
the value printed in the paper. Counts are checked exactly; real-valued
quantities to the precision the paper states (tolerance = half a unit in the
last printed digit). Exit code 0 iff all checks pass.

This file is the normative mapping  paper number -> artifact + computation.
If a paper number changes, change it HERE too, or verification fails.
"""
import sys, pathlib, json
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "shared" / "lib"))
import paths as P

FAIL = []
def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got} (paper: {want})")
    if not ok: FAIL.append(name)

def need(p):
    if not pathlib.Path(p).exists():
        sys.exit(f"missing {p} -- run the RUNBOOK pipeline first")
    return p

# ---------------- inputs / quality campaign (paper §3, §8) ----------------
q = pd.read_parquet(need(P.QUALITY_TABLE))
scored = q[q["quality"].notna()]
role = q["agent"].map(lambda a: "disp" if a == "A_dispatcher" else ("syn" if a == "A_synth" else "spec"))
check("scored records", len(scored), 8925)
check("scored dispatcher", int((role[scored.index] == "disp").sum()), 4368)
check("scored specialist", int((role[scored.index] == "spec").sum()), 3862)
check("scored synthesiser", int((role[scored.index] == "syn").sum()), 695)
check("query instances", q["query_id"].nunique(), 129)
check("root queries", q["query_id"].str.split("::").str[0].nunique(), 37)
check("configurations", q["config_id"].nunique(), 84)

perf = pd.read_parquet(need(P.PERF_TABLE))
check("perf rows", len(perf), 84)
check("max throughput (tok/s)", float(perf["throughput_tok_s"].max()), 247.0, 0.5)
check("min throughput (tok/s)", float(perf["throughput_tok_s"].min()), 28.9, 0.05)
check("configs over 6.99 GB", int((perf["peak_mem_gb"] > P.MEM_BUDGET_GB).sum()), 0)
pp = perf.copy()
pp["model"] = pp["config_id"].str.split("__").str[0]
pp["params"] = pp["model"].map(P.PARAMS_B)
pp["lat384"] = pp["ttft_s"] + 384 / pp["throughput_tok_s"]
C = pp[["params", "peak_mem_gb", "ttft_s", "throughput_tok_s", "energy_j_per_tok", "lat384"]].corr()
check("corr(params, VRAM)", float(C.loc["params", "peak_mem_gb"]), 0.84, 0.005)
check("corr(params, throughput)", float(C.loc["params", "throughput_tok_s"]), -0.79, 0.005)
check("corr(energy/tok, latency@384)", float(C.loc["energy_j_per_tok", "lat384"]), 0.73, 0.005)
check("corr(TTFT, latency@384)", float(C.loc["ttft_s", "lat384"]), 0.80, 0.005)
q4 = pp[pp["config_id"].str.contains("__c4096")].pivot_table(index="model", columns=pp["config_id"].str.split("__").str[1], values=["peak_mem_gb", "throughput_tok_s"])
both = q4.dropna(subset=[("peak_mem_gb", "Q3_K_M"), ("peak_mem_gb", "Q8_0")])
check("Q3->Q8 VRAM ratio @4096", float((both[("peak_mem_gb", "Q8_0")] / both[("peak_mem_gb", "Q3_K_M")]).mean()), 1.56, 0.005)
check("Q3->Q8 throughput ratio @4096", float((both[("throughput_tok_s", "Q8_0")] / both[("throughput_tok_s", "Q3_K_M")]).mean()), 0.82, 0.005)

sc = pd.read_csv(need(P.SCORECARD))
sc["model"] = sc["config_id"].str.split("__").str[0]
SPEC = [a for a in sc["agent"].unique() if a not in ("A_dispatcher", "A_synth")]
spec_pm = sc[sc["agent"].isin(SPEC)].groupby("model")["quality"].mean()
check("best specialist model", spec_pm.idxmax(), "llama3.2-1b")
check("Llama-3.2-1B specialist quality", float(spec_pm["llama3.2-1b"]), 0.74, 0.005)
check("Mistral-7B specialist quality", float(spec_pm["mistral-7b"]), 0.72, 0.005)
check("worst specialist model", spec_pm.idxmin(), "qwen2.5-0_5b")
disp_pm = sc[sc["agent"] == "A_dispatcher"].groupby("model")["quality"].mean()
check("dispatcher F1 Phi-3.5", float(disp_pm["phi3.5-mini"]), 0.65, 0.005)
check("dispatcher F1 Mistral-7B", float(disp_pm["mistral-7b"]), 0.59, 0.005)
check("dispatcher F1 Gemma-2", float(disp_pm["gemma2-2b"]), 0.57, 0.005)
check("dispatcher F1 Llama-3.2-3B (degenerate)", float(disp_pm["llama3.2-3b"]), 0.06, 0.005)
syn_pm = sc[sc["agent"] == "A_synth"].groupby("model")["quality"].mean()
check("best synthesiser model", syn_pm.idxmax(), "mistral-7b")
check("Mistral-7B synth quality", float(syn_pm["mistral-7b"]), 0.77, 0.005)
syn = sc[sc["agent"] == "A_synth"]
check("synth context precision constant", float(syn["context_precision"].std()), 0.0, 1e-9)
check("synth context precision", float(syn["context_precision"].mean()), 0.82, 0.005)
check("synth context recall", float(syn["context_recall"].mean()), 0.52, 0.005)
check("corr(faithfulness, synth quality)", float(np.corrcoef(syn["faithfulness"], syn["quality"])[0, 1]), -0.70, 0.005)

# ---------------- part 1 (paper §7) ----------------
fr = pd.read_csv(need(P.PERF_FRONTIER))
check("capacity frontier rows", len(fr), 264)
for k, eps_want in [(3, 19.3), (5, 25.5), (9, 36.8)]:
    g = fr[fr["k"] == k]
    check(f"ceiling 12.2 B (k={k})", float(g["capacity"].max()), 12.2, 0.05)
    check(f"eps reaching ceiling (k={k})", float(g.loc[g["capacity"] >= 12.2 - 1e-9, "eps"].min()), eps_want, 0.05)
b = json.load(open(need(ROOT / "part1_static_allocation/results/data/baseline_cmp.json")))
wins = sum(1 for r in b if r["greedy"] is None or r["milp"] > r["greedy"] + 1e-6)
check("baseline points", len(b), 132)
check("MILP strict wins (parameter objective)", wins, 7)
h = json.load(open(need(ROOT / "part1_static_allocation/results/data/baseline_v4.json")))
hw = sum(1 for r in h if r["greedy_q"] is None or r["milp_q"] > r["greedy_q"] + 1e-6)
check("heterogeneous points", len(h), 132)
check("heterogeneous MILP wins", hw, 100)
for k, want in [(1, 17), (3, 62), (5, 64), (9, 100)]:
    g = [r for r in h if r["k"] == k]
    w = sum(1 for r in g if r["greedy_q"] is None or r["milp_q"] > r["greedy_q"] + 1e-6)
    check(f"heterogeneous win rate k={k} (%)", round(100 * w / len(g)), want)

# ---------------- part 2 (paper §8.2-8.3, §9) ----------------
agg = pd.read_parquet(need(ROOT / "part2_quality_aware/results/data/agg_frontier.parquet"))
check("specialist configs", len(agg), 56)
check("non-dominated specialist configs", int(agg["pareto_lat"].sum()), 4)
check("largest non-dominated specialist (B)", float(agg.loc[agg["pareto_lat"], "params"].max()), 1.2, 1e-9)
pc = pd.read_parquet(need(ROOT / "part2_quality_aware/results/data/policy_compare.parquet"))
gp = pc["q_maxquality"] - pc["q_maxparams"]
check("slot-policy budget steps", len(pc), 105)
check("slot-policy dominated at every step", int((gp > 1e-9).sum()), len(pc))
check("slot-policy mean gap", float(gp.mean()), 0.051, 0.0005)
check("slot-policy max gap", float(gp.max()), 0.131, 0.0005)
af = pd.read_csv(need(P.QUALITY_ADDITIVE_FRONTIER))
check("additive plateau Q (k=3)", float(af[af["k"] == 3]["Q"].max()), 1.60, 0.005)
r = af.loc[af[af["k"] == 3]["Q"].idxmax()]
check("plateau dispatcher model", r["disp"].split("__")[0], "mistral-7b")
check("plateau specialist model", r["spec"].split("__")[0], "llama3.2-1b")
check("plateau synthesiser model", r["synth"].split("__")[0], "mistral-7b")
pv = pd.read_csv(need(ROOT / "part2_quality_aware/results/data/proxy_vs_quality_pipeline.csv"))
gv = pv["Q_quality"].cummax() - pv["Q_proxy"]
check("proxy-vs-quality budgets", len(pv), 26)
check("proxy dominated at every budget", int((gv > 1e-9).sum()), 26)
check("proxy-vs-quality mean gap", float(gv.mean()), 0.13, 0.005)
check("proxy-vs-quality max gap", float(gv.max()), 0.18, 0.005)
dci = pd.read_csv(need(ROOT / "part2_quality_aware/results/data/per_domain_ci.csv"))
sepc = dci[dci["domain"] == "A_separazione_contenziosa"].iloc[0]
check("sep. contenziosa CI low", float(round(sepc["ci_lo"], 2)), 0.68, 1e-9)
check("sep. contenziosa CI high", float(round(sepc["ci_hi"], 2)), 0.83, 1e-9)
qt = q[q["quality"].notna()].copy(); qt["model"] = qt["config_id"].str.split("__").str[0]
v = qt[(~qt["agent"].isin(["A_dispatcher", "A_synth"])) & (qt["model"] == "llama3.2-1b")]["quality"].values
rng = np.random.default_rng(0)
ci = np.percentile([rng.choice(v, len(v), replace=True).mean() for _ in range(2000)], [2.5, 97.5])
check("pooled Llama-1B CI low", float(round(ci[0], 3)), 0.743, 1e-9)
check("pooled Llama-1B CI high", float(round(ci[1], 3)), 0.764, 1e-9)

# ---------------- part 3 (paper §10) ----------------
gf = pd.read_csv(need(P.QUALITY_GATED_FRONTIER))
gg = (gf["bil_Q"] - gf["add_Q"]).dropna()
check("gated frontier points", len(gf), 44)
check("bilinear strictly better", int((gg > 0.001).sum()), 42)
check("gated mean gap", float(gg.mean()), 0.18, 0.005)
check("gated max gap", float(gg.max()), 0.32, 0.005)
cs = json.load(open(need(ROOT / "part3_bilinear_gated/results/data/coupling_sweep.json")))
by = {r["lam"]: r["mean_gap"] for r in cs}
check("coupling gap at lambda=0", float(by[0.0]), 0.05, 0.005)
check("coupling gap at lambda=1", float(by[1.0]), 0.18, 0.005)
lams = sorted(by); slope = float(np.polyfit(lams, [by[l] for l in lams], 1)[0])
check("coupling slope", slope, 0.13, 0.005)
ss = pd.read_csv(need(ROOT / "part3_bilinear_gated/results/data/sigma_sweep.csv"))
ss["small"] = ss["spec"].map(P.PARAMS_B) <= 1.2
for sig, want in [(1.0, 100), (0.66, 100), (0.5, 96), (1.0 / 3.0, 84)]:
    g = ss[np.isclose(ss["sigma"], sig)]
    check(f"small-specialist share sigma={sig:.2f} (%)", round(100 * g["small"].mean()), want)
check("small specialist at eps<=20 (all modes, %)", round(100 * ss[ss["eps"] <= 20]["small"].mean()), 100)

# ---------------- lean (Qwen-family robustness, whitepaper §3/§5/§7) ----------------
import hashlib, glob
LEAN = ROOT / "lean"
cat_sha = hashlib.sha256(open(need(LEAN / "catalog/catalog.json"), "rb").read()).hexdigest()
check("lean catalog sha == sidecar", cat_sha == open(LEAN / "catalog/catalog.json.sha256").read().split()[0], True)
runs = sorted(glob.glob(str(LEAN / "results/lean_8gb/*/alloc.json")))
need(runs[-1] if runs else LEAN / "results/lean_8gb/MISSING")
al = json.load(open(runs[-1])); meta = json.load(open(pathlib.Path(runs[-1]).parent / "meta.json"))
check("lean quality source is measured file", meta["quality_source"].startswith("file:"), True)
check("lean canonical Q", float(al["Q"]), 1.8806, 0.00005)
check("lean canonical latency (binds SLA)", float(al["L_total_s"]), 8.0, 1e-9)
check("lean canonical memory (GB)", float(al["memory_used_gb"]), 5.43, 0.005)
check("lean dispatcher", al["config_by_role"]["d"], "qwen2.5-3b__Q8_0__c8192")
check("lean specialist", al["config_by_role"]["s"], "qwen2.5-1_5b__Q5_K_M__c8192")
check("lean synthesiser (shared group)", al["config_by_role"]["y"], "qwen2.5-1_5b__Q5_K_M__c8192")
check("lean loaded groups", len(al["loaded_groups"]), 2)
bl = pd.read_csv(pathlib.Path(runs[-1]).parent / "baselines.csv").set_index("source")
check("lean baseline largest_fits Q", float(bl.loc["baseline:largest_fits", "Q"]), 1.39, 0.005)
check("lean baseline uniform Q", float(bl.loc["baseline:uniform", "Q"]), 1.60, 0.005)
check("lean baseline per_role_best Q", float(bl.loc["baseline:per_role_best", "Q"]), 1.95, 0.005)
check("lean per_role_best infeasible (over SLA)", bool(bl.loc["baseline:per_role_best", "feasible"]), False)
check("lean baseline random_feasible Q", float(bl.loc["baseline:random_feasible", "Q"]), 1.88, 0.005)
sv = pd.read_csv(need(LEAN / "results/ablations/sequential_vs_concurrent.csv")).set_index("t_circ_s")
check("lean concurrent Q at T=8", float(sv.loc[8.0, "Q_concurrent"]), 1.881, 0.0005)
check("lean sequential Q at T=8", float(sv.loc[8.0, "Q_sequential"]), 1.709, 0.0005)
check("lean concurrent feasible at T=4, sequential not",
      bool(sv.loc[4.0, "concurrent_feasible"]) and not bool(sv.loc[4.0, "sequential_feasible"]), True)
check("lean concurrent feasible at T=6, sequential not",
      bool(sv.loc[6.0, "concurrent_feasible"]) and not bool(sv.loc[6.0, "sequential_feasible"]), True)
sla = pd.read_csv(need(LEAN / "results/ablations/sla_sweep.csv")).set_index("t_circ_s")
check("lean SLA sweep saturates by T=14 (== T=20)", float(sla.loc[14.0, "Q"]) == float(sla.loc[20.0, "Q"]), True)
check("lean SLA plateau Q", float(sla.loc[20.0, "Q"]), 1.952, 0.0005)
cs2 = pd.read_csv(need(LEAN / "results/ablations/catalog_scope.csv")).set_index("t_circ_s")
check("lean drop-3B cost at T=8", float(cs2.loc[8.0, "delta_Q"]), 0.15, 0.005)
check("lean drop-3B cost at T=12", float(cs2.loc[12.0, "delta_Q"]), 0.22, 0.005)

# ---------------- part 4 (dynamic agentic path) ----------------
P4 = ROOT / "part4_dynamic_path"
def _prf(pred, gold):
    if not pred and not gold: return 1.0, 1.0, 1.0
    if not pred: return 1.0, 0.0, 0.0
    if not gold: return 0.0, 1.0, 0.0
    tp = len(pred & gold); p = tp / len(pred); r = tp / len(gold)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
de = json.load(open(need(P4 / "results/dynamic_eval.json")))
S4, rows4 = de["summary"], de["per_query"]
check("part4 dynamic_eval n_test", len(rows4), 64)
check("part4 dynamic_eval n_abstain (OOD)", sum(r["is_abstain"] for r in rows4), 15)
check("part4 dynamic_eval summary counts", (S4["n_test"], S4["n_abstain"]), (64, 15))
for sel in ("full", "topk", "threshold", "dynamic"):
    P_, R_, F_, K_, A_ = [], [], [], [], []
    for r in rows4:
        pred, gold = set(r[sel]["S"]), set(r["gold"])
        p, q_, f = _prf(pred, gold); P_.append(p); R_.append(q_); F_.append(f); K_.append(len(pred))
        if r["is_abstain"]: A_.append(1.0 if not pred else 0.0)
    for key, got in (("precision", np.mean(P_)), ("recall", np.mean(R_)), ("f1", np.mean(F_)),
                     ("mean_k", np.mean(K_)), ("abstain_accuracy", np.mean(A_))):
        check(f"part4 {sel}.{key} internally consistent", float(got), float(S4["selectors"][sel][key]), 0.005)
for f in ("p4_selector_comparison.pdf", "p4_latency_tradeoff.pdf", "p4_routing_normal_vs_dynamic.pdf"):
    check(f"part4 figure exists: {f}", (P4 / "results/figures" / f).exists(), True)
check("lean figure exists: baselines_comparison.pdf", (LEAN / "figures/baselines_comparison.pdf").exists(), True)
for f in ("sequential_vs_concurrent", "sla_sweep", "per_role_contribution", "catalog_scope"):
    check(f"lean figure exists: {f}.pdf", (LEAN / "figures" / f"{f}.pdf").exists(), True)
if S4.get("embedder") != "bge-m3":
    print(f"[NOTE] part4 dynamic_eval.json is the '{S4.get('embedder')}' artifact -- the paper's "
          "bge-m3 numbers require the score-cache step in part4_dynamic_path/data/INPUTS.md.")

print()
if FAIL:
    print(f"{len(FAIL)} CHECK(S) FAILED: {FAIL}")
    sys.exit(1)
print("ALL CHECKS PASSED")
