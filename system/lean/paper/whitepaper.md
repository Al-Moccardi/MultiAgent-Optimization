# Quality-aware MAMAP — a lean formulation

**Reference doc for the paper.** Sits alongside the colleague's three parts in
`mamap_repo/` and shrinks the formulation to a single canonical algorithm
that is easy to write down, reproduce, and ablate.

---

## 1. Algorithm box

**Sets.** Roles $r \in \{d, s, y\}$ (dispatcher, specialist, synthesiser).
Configs $c = (m_c, q_c, n_c) \in \mathcal{K}$ — a model–quantisation–context
triple. Groups $g = (m, q) \in \mathcal{G}$ with the projection
$g(c) = (m_c, q_c)$. Domains $D$ — topical labels (medical / finance / …)
indexing the specialist's quality scores.

**Decision variables.**
- $x_{r,c} \in \{0,1\}$ — role $r$ uses config $c$ (sparse over $c \in \mathcal{K}_r$).
- $z_g \in \{0,1\}$ — group $g$ is loaded.
- $\Lambda \in \mathbb{R}_{\ge 0}$ — specialist latency under concurrent serving.

**Objective.**

$$
\max_{x,\,z,\,\Lambda}\quad
Q(x) \;=\;
F_d\!\big(c_d\big)
\;+\; \frac{1}{|D|}\sum_{\delta \in D} Q_s\!\big(c_s, \delta\big)
\;+\; Q_y\!\big(c_y\big)
$$

$F_d, Q_s, Q_y \in [0, 1]$ are measured on the chosen RAG benchmark
(`rag_bench/eval.py`, RAGAS-aggregate via
`mamap_repo/shared/faiss_code/scorer.py`). **No scalarisation weights** —
every term is a unit-interval score.

**Constraints.**
- Assignment: $\sum_{c \in \mathcal{K}_r} x_{r,c} = 1$ for every $r$.
- Load–use: $x_{r,c} \le z_{g(c)}$ for every $r, c$.
- Memory (hard): $\sum_g w_g\,z_g + \sum_{r,c} \kappa_c\,x_{r,c} \le M$
  with $\kappa_c = 2\,L\,n_{kv}\,d\,n_c\,b_{kv}$ (closed form).
- Per-group concurrent latency (cover): $\Lambda \ge L_{s,c} \cdot x_{s,c}$ for every
  $c \in \mathcal{K}_s$.
- SLA (hard): $L_d(c_d) + \Lambda + L_y(c_y) \le T^\circ$.

**Eligibility.** Per-role $\mathcal{K}_r = \{c : L_c \le T^\circ\}$ is
pre-filtered. The MILP therefore never sees SLA-violating rows, which is
both cleaner to write down and yields smaller models.

The MILP is solved exactly with HiGHS (`highspy`); deterministic with
`random_seed=0` and single-threaded.

---

## 2. Catalog (paper-clean, narrow on purpose)

| | |
|---|---|
| Family | Qwen2.5 only |
| Sizes | 0.5 B, 1.5 B, 3 B |
| Quants | Q3_K_M, Q5_K_M, Q8_0 |
| Contexts | 2048, 4096, 8192 |
| Total raw triples | $3 \times 3 \times 3 = 27$ |
| Feasible at $M = 6.99\,\mathrm{GB}$ | $\approx 20$ (varies with $T^\circ$) |

This narrowing addresses the user's comments:
- **C5** (drop the MoE outlier and shrink the search space): only one
  family, single dense architecture, three sizes spanning a $6\times$ range.
- **C2** (selection mechanism made explicit): no greedy pre-filter —
  *every* feasible triple is a MILP decision variable. The pruning is
  catalog construction itself, transparent in `lean/catalog/catalog.yaml`.
- **Reproducibility:** `lean/catalog/build_catalog.py` joins
  `lean/catalog/catalog.yaml` with the colleague's measured
  `mamap_repo/shared/data/perf_table.parquet`. Output `catalog.json` carries
  a SHA-256 sidecar.

---

## 3. The concurrent latency model (comment **C4**)

The colleague's parts/1–3 use **strictly sequential**:

$$L_\text{total}^\text{seq} \;=\; L_d + k\cdot L_s + L_y.$$

This is conservative — it assumes each of $k$ activated specialists in a
query is run end-to-end before the next starts.

The lean version uses **per-group concurrent**:

$$L_\text{total}^\text{conc} \;=\; L_d + \Lambda + L_y, \qquad \Lambda \ge L_{s, c}\,x_{s,c}\;\;\forall c.$$

The single $\Lambda$ collapses to the specialist latency on the chosen
config under the assumption that specialists sharing the loaded $(m, q)$
group execute in a single batched pass (matches vLLM-style concurrent
serving). Since at most one $x_{s,c} = 1$, $\Lambda$ is exact; the
formulation needs one continuous auxiliary plus $|\mathcal{K}_s|$ cover rows.

### Ablation: sequential vs concurrent (synthetic Q, $k_\text{active} = 3$)

See `results/ablations/sequential_vs_concurrent.csv` / `figures/sequential_vs_concurrent.pdf`.

| $T^\circ$ (s) | $Q^\star_\text{conc}$ | $Q^\star_\text{seq}$ | concurrent feasible? | sequential feasible? |
|---:|---:|---:|:---:|:---:|
| 4 | 1.420 | — | ✓ | ✗ |
| 6 | 1.529 | — | ✓ | ✗ |
| 8 | 1.559 | 1.450 | ✓ | ✓ |
| 10 | 1.574 | 1.480 | ✓ | ✓ |
| 12 | 1.598 | 1.529 | ✓ | ✓ |
| 15 | 1.598 | 1.559 | ✓ | ✓ |

Concurrent dominates everywhere it overlaps, and stays feasible at
$T^\circ$ values where the sequential model rejects the instance. The
margin is exactly the relaxed slack $(k_\text{active} - 1)\cdot L_s$ at the
binding constraint.

---

## 4. Naive baselines (comment **C3**)

The four shapes from main MAMAP, role-aware:

| Baseline | Mechanism |
|---|---|
| **largest-fits** | Pick the maximum-params $(m,q)$ that fits in $M$ and respects $T^\circ$ across all three roles using one shared group; replicate. |
| **per-role-best** | For each role independently pick $\arg\max_c Q_r(c)$ in $\mathcal{K}_r$ ignoring shared memory. May overflow $M$ (report). |
| **uniform** | Same config across all roles; pick the one maximising total $Q$ that fits. |
| **random-feasible** | Sample 500 feasible $(x, z)$ via rejection; keep the best. |

The MILP is provably an upper bound — `tests/test_baselines.py::test_milp_beats_or_matches_all_baselines` asserts this.

---

## 5. Ablations

| § | Question | Output |
|---|---|---|
| 5.1 | Does the concurrent latency model materially help over the colleague's sequential? | `sequential_vs_concurrent.{csv,pdf}` — yes; see §3. |
| 5.2 | How does $Q^\star$ scale with the SLA budget $T^\circ$? | `sla_sweep.{csv,pdf}` — monotone in $T^\circ$, saturates around 14 s. |
| 5.3 | Which role drives $Q$? | `per_role_contribution.{csv,pdf}` — locks two roles to their min-memory configs, varies the third. The joint MILP exceeds the lock-2/vary-1 estimate when sharing a group frees memory for a richer config elsewhere. |
| 5.4 | What is the marginal value of the 3B tier? | `catalog_scope.{csv,pdf}` — dropping 3B costs $\approx 0.03$ Q at most $T^\circ$, growing to $\approx 0.05$ at $T^\circ = 12$ s. |
| 5.5 (deferred) | Match colleague's parts/1–3 on identical input. | Out of scope here: the colleague's RAGAS scores are on a different (Italian family-law) dataset; cross-applying them would be misleading. Future work. |

---

## 6. Reproducibility

Every `lean/results/<run_id>/meta.json` carries:
- `catalog_sha256` — a hash of the materialised catalog rows
  (model, quant, context, ttft, throughput, energy per token).
- `master_seed`, `quality_source` ("synthetic" | "file:quality.parquet").
- `platform`, `python`, `pkg_versions` (`highspy`, `pydantic`, `numpy`,
  `pandas`).
- `timings_s` for every solver call.
- The MILP allocation (`milp.config_by_role`, `milp.Q`, `milp.L_total_s`,
  `milp.memory_used_gb`) and each baseline's row.

`tests/test_experiment.py::test_run_one_is_deterministic_at_same_seed`
asserts that two seeded reruns produce identical MILP optima.

---

## 7. Quality data — current status

The lean MILP and every ablation in §5 are currently driven by
**`lean/catalog/quality.parquet`**, produced by
`scripts/import_colleague_quality.py` from the colleague's
`shared/data/quality_table.parquet`. Concretely:

- The colleague evaluated their 11 agents (1 dispatcher, 9 legal
  specialists, 1 synthesiser) on 52 Italian family-law queries with each
  of their 27 Qwen2.5 catalog rows (sizes 0.5 B / 1.5 B / 3 B, quants
  Q3_K_M / Q5_K_M / Q8_0, contexts 2k / 4k / 8k).
- For role `d` and `y` we take **mean(quality)** across all queries the
  agent was scored on.
- For role `s` we take **per-(config, specialist) mean** restricted to
  queries the specialist was *activated* on — the colleague's
  ground-truth for "this specialist's quality on its domain".
- Specialist domains are the 9 colleague-defined legal categories
  (`succ_testamentaria`, `separazione_consensuale`, …).

A NaN-aware coverage filter (`src/quality_coverage.py`) **excludes from
$\mathcal{K}_r$** any config the colleague did not measure for that role.
Important consequence: the colleague evaluated `A_synth` only at
ctx=8192, so $\mathcal{K}_y$ contains 9 configs (one per size × quant).
This is faithfully propagated: the MILP can only choose synthesiser
configs the colleague actually scored.

**Provenance.** Every solver run records `meta.json` with
`quality_source = "file:quality.parquet"` plus the catalog SHA-256;
`quality.meta.json` (sibling of the parquet) records the source file
SHA-256, the agent→role/domain mapping, and per-key sample sizes.

A **synthetic fallback** (`src.quality.synthetic_quality(...)`) remains
shipped as the CI default — it encodes the same qualitative shape (the
colleague's reported "specialist quality is non-monotone, peaks at
~1.5 B" finding is reproduced) but never used for paper numbers.

## 7.1 What real-quality optimization picks

On the canonical `lean_8gb.yaml` ($M = 6.99$ GB, $T^\circ = 8$ s):

- **Dispatcher**: `qwen2.5-3b__Q8_0__c8192` (highest measured F1).
- **Specialist**: `qwen2.5-1_5b__Q5_K_M__c8192` — *not* the largest
  model. Matches the colleague's headline finding that specialist
  quality is non-monotone in size and peaks at the smaller end.
- **Synthesiser**: `qwen2.5-1_5b__Q5_K_M__c8192` (shared with
  specialist).
- **Loaded groups**: 2 (3 B Q8 + 1.5 B Q5). The MILP exploits sharing —
  loading the 1.5 B model once for both specialist and synth roles
  leaves room for the much heavier 3 B Q8 dispatcher.
- $Q = 1.881$, memory $= 5.43$ GB, latency $= 8.0$ s (binds the SLA).

Baselines (same instance):

| Baseline | Q | L (s) | Mem (GB) | Feasible? |
|---|---:|---:|---:|:---:|
| `largest_fits` | 1.39 | 6.34 | 1.72 | ✓ |
| `uniform` | 1.60 | 6.84 | 2.00 | ✓ |
| `per_role_best` | 1.95 | 10.97 | 6.37 | ✗ (over SLA + over mem) |
| `random_feasible` | 1.88 | 6.74 | 6.36 | ✓ |
| **MILP** | **1.88** | **8.0** | 5.43 | ✓ |

---

## 8. Limitations + future work

- **Synthetic quality** until the rag_bench harness is exercised on a
  real domain-labelled benchmark.
- **Concurrent latency** assumes vLLM-style concurrent serving for
  specialists sharing $(m, q)$. Documented; sequential MILP is shipped
  as an ablation.
- **Single device.** The lean formulation is per-device; sharing across
  multiple cards / nodes is out of scope.
- **Static allocation.** The downstream **dynamic optimization** step
  (per-query adaptation, request-time re-routing) consumes the typed
  `Allocation` artifact written to `meta.json` and is out of scope for
  this package.

---

## 9. Pointers

- Algorithm: `src/milp.py` (HiGHS), `src/latency.py` (per-group concurrent
  cover).
- Eligibility + per-role $\mathcal{K}_r$: `src/instance.py`.
- Baselines: `src/baselines.py`.
- Experiment runner: `src/experiment.py`.
- Ablations: `src/ablations.py` (four sweeps as standalone functions).
- Figures: `src/figures.py`.
- rag_bench harness: `rag_bench/` (build_subset.py, eval.py, README.md).
- Tests: `tests/` (55 passing, lint-clean).
