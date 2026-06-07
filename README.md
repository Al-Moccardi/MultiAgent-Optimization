# Quality Aware Multi Agent Placement System

**Quality-Aware Multi-Agent Placement for on-device RAG.** An 11-agent Italian
family-law RAG pipeline — one routing dispatcher, nine legal-domain
specialists (succession, separation, ADR, minor protection, matrimonial
property, …) and one synthesiser — must run on a single consumer GPU
(RTX 4070 Laptop, 8 GB, 6.99 GB usable). Each role must be assigned a
(model × quantization × context) configuration from a zoo of 10 dense SLMs
(0.36–7.2 B; Qwen2.5, Llama-3.2, SmolLM2, Gemma-2, Phi-3.5, Mistral),
75 measured configurations in total, under a hard memory budget and a
latency target. The paper formulates this allocation as a MILP and studies
what the *measured* problem actually looks like.

## What the paper shows (and where)

1. **Measure once, optimise many** (§5). Per-configuration latency, memory
   and energy are measured once on the target device
   (`shared/data/perf_table.parquet`); every optimisation below then re-solves
   in seconds on the frozen tables, with no further hardware access.
2. **Capacity allocation is easy; capacity is the wrong objective** (§6–§7).
   A parameter-capacity MILP beats greedy/uniform baselines (84/91 budgets),
   but §8 shows measured quality does not track parameter count
   (r = 0.54/0.31, non-monotone): the best specialist is Llama-3.2-1B (0.74),
   ahead of Mistral-7B at 6× the size, and only 4/50 specialist
   configurations are Pareto-optimal in quality–latency, none above 1.2 B.
   A leave-one-domain-out protocol shows this is not selection overfitting.
3. **Quality-aware allocation has a three-regime structure** (§9): small
   specialist, large dispatcher, large synthesiser — the near-opposite of the
   capacity optimum, worth up to 0.18 quality at matched budgets.
4. **Router–specialist coupling matters** (§10): a gated bilinear objective
   (McCormick-linearised at model level) picks a *different*, dominating
   dispatcher in 42/44 budgets — the low-global-F1 / high-activated-recall
   SmolLM2-360M — versus the coupling-blind additive model.
5. **The execution model moves the frontier, not the optimum** (§10.5):
   exact concurrent serving (one auxiliary Λ) enlarges feasibility
   (4 s vs 7 s) but the three-regime allocation is invariant.
6. **Both quality findings are cross-family phenomena** (§11): restricted to
   a single family (Qwen2.5), the small specialist becomes a budget artifact
   (switches to 3 B at ε = 11 s) and the gated/additive gap collapses to
   zero (33/33 and 37/37 budgets) — pool diversity is what is load-bearing.
7. **A dynamic routing layer on top of the static allocation** (§12):
   a calibrated, LLM-free gate over frozen BGE-M3 domain profiles routes
   k = 3.61 specialists per query on average (vs static k = 6), cutting
   latency 9.1–9.3 % concurrent / 27–35 % sequential at equal-or-better
   routing F1, abstaining on 9/15 out-of-distribution probes.

## Repository layout

```
paper/          mamap_full.tex + fig/ + mamap_full.pdf (51 pp) and
                AUDIT_NOTES.md, the claim-by-claim audit ledger.
optimization/   the optimisation study (§5–§11): measurement tables in
                shared/data (75 configs, 10 dense models), the
                part1/part2/part3 experiment scripts behind every §5–§10
                figure and number, shared/lib (incl. lexsolve.py, the
                canonical tie-break), and scripts_verify/ — four standalone
                scripts that re-derive the §8/§10.5/§11.1/§11.2 statistics
                and print each measured value against the paper's expected
                value.
system/         the deployed measurement + inference system and the §12
                dynamic layer (part4_dynamic_path), with the frozen score
                cache, the domain-profile vector sidecar, the OOD manifest
                and the verified §12 result JSONs committed under
                part4_dynamic_path/.
```

`optimization/` and `system/` are two snapshots of the same project with
incompatible path layouts (`shared/lib/paths.py` differs); merging them would
silently break the verified runs, so they ship side by side, self-contained.

## Quick start (reviewer)

```bash
pip install -r requirements.txt          # CPU-only, CBC ships with pulp
cd optimization/scripts_verify
python verify_sec8_stats.py              # §8  statistics
python verify_sec10_concurrent.py        # §10.5 seq vs exact concurrent
python verify_sec11_family.py            # §11.1 single-family ablation
python verify_sec11_gated_collapse.py    # §11.2 gated/additive collapse
```

Each script prints `measured (expect X)` per claim; a clean run shows every
pair equal. `VERIFY.md` is the full claim → command → artifact map for the
whole paper, including the §12 replay commands (`system/requirements.txt`
covers the GPU embedding stack; replay from the committed cache is CPU-only).

## Known, deliberate retentions

The excluded MoE model (`ministral-3b`, a granite-3.1-3b-a800m repack) is
removed from all catalogs and from the optimisation tables (75 configs,
10 dense models). Two documented exceptions (VERIFY.md §C): inert
`ministral-3b` keys in some scripts' parameter-lookup dicts (reruns proven
unaffected), and `system/part4_dynamic_path/data/quality_table.parquet`,
which ships byte-identical to what the verified §12 runs consumed because
`specialist_quality_by_domain` has a pooled-over-all-configs fallback, so
filtering it is not provably inert.

`system/AUTHOR_TODO.md` lists the single remaining author-side artifact
(`routing_eval_concurrent.json`).
