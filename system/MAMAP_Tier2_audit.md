# MAMAP-Edge — Tier 2 audit (modelling consistency & objective validity)

**Scope.** Tier 1 was a provable arithmetic bug (the ε-grid scale) with a single
correct fix, already patched. Tier 2 is different in kind: the code does what it
intends, but the *model* either (a) contradicts its own premises, (b) implements a
defended choice that nonetheless hides a real gap, or (c) optimises a quantity
whose validity is questionable. **None of these are bugs to silently "fix."** Each
is a modelling decision for the author. This document states each finding, the
strongest defence of the current design, what survives scrutiny, and the
research-correct resolution — separating what is a *code option* from what is a
*paper argument* from what is a *dataset limitation*.

Every quantitative claim here was checked against the source in the uploaded
codebase, not asserted from memory. Two earlier Tier-1/Tier-2 claims were
**withdrawn** after that checking (noted below); they are documented so the
reasoning trail is honest.

---

## Summary table

| # | Finding | Status | Resolution register |
|---|---------|--------|---------------------|
| 1 | Q³ "double-counts router recall" | **WITHDRAWN** (math refuted it) | none — Q³ is a legitimate joint expectation |
| 3 | Latency worst-case vs gated quality | **Defended choice, but its proposed alternative is physically wrong** | code option (correct variant) + paper note |
| 4 | Objective is an unnormalised sum of incommensurable terms | **Real weighting distortion** | code option (knob) + paper argument |
| 4′ | Routing **precision** is under-incentivised | **Real, precise** | paper argument (+ optional energy/latency penalty) |
| 5 | Specialist Q excludes correctness | **Defended tradeoff with a real validity cost** | dataset change + paper argument |
| 6 | Shared-embedder circularity (retriever == scorer) | **Real validity threat** | measurement change + paper caveat |
| 7 | Cosine metrics: compressed range, regurgitation reward | **Real** | paper caveat + human-subset validation |
| 8 | Several specialists have N_s ≤ 3 | **Real** | dataset + bootstrap (post-hoc) |

---

## Finding 1 — WITHDRAWN: Q³ does *not* double-count recall

**Original claim.** `Q³[s,k_a,k_d] = (1/N_s)·Σ_q 1[s∈pred(k_d,q)]·Q_gen` divides by
`N_s = |expected(s)|` while the numerator only sums over fired queries, so a
low-recall router scales every specialist down — and recall already appears in the
standalone router F1 term, hence "double-counted."

**Why it was withdrawn.** Decompose with `H_s(k_d) = {q expected ∧ fired}`:

```
Q³ = (|H_s(k_d)|/N_s) · mean_{q∈H_s} Q_gen = recall_s(k_d) · E[Q_gen | fired]
```

This is the **joint expectation** `E_q[1[fired]·Q_gen]` — the expected per-query
quality contribution of `s`, correctly accounting for the chance the router drops
it. It is well-defined, not a normalisation error. The recall-weighting does
*necessary* work: given two routers with **equal F1**, it correctly prefers the
higher-recall one (verified numerically: identical specialist earns 0.90 under
recall 1.0 vs 0.30 under recall 0.33). "Fixing" it to the conditional
`E[Q_gen|fired]` (dividing by `|H_s|`) would **delete** the downstream-activation
signal the bilinear term exists to capture — the opposite of correct.

**What remains real** is *labelling*: v3 eq. (3)'s prose calls Q³ "conditional,"
but the formula (which governs) is the *joint*. One-word paper fix: say "joint
expected contribution," not "conditional expected quality." No code change.

---

## Finding 3 — Worst-case latency is defended; its proposed *alternative* is physically wrong

**Current code** (`milp.py`): `L_max_spec ≥ Σ_k L[s,k]·x[s,k]` for **every**
specialist → `L_max_spec` is the slowest of all nine, regardless of routing. The
objective, by contrast, credits only router-activated specialists.

**Steelman (and it holds).** v3 §9.1 *explicitly* makes this choice and defends it:
*"We chose worst-case because legal-assistant SLOs should hold uniformly."* If any
query could fan out to all nine, and specialists run in parallel, the slowest
*loaded* specialist is the true worst-case wall-clock. The constraint is therefore
**correct and intentional**, not an oversight. It should **not** be overwritten.

**The real gap (new — the paper does not see this).** §9.1 offers a "straightforward
substitute": an expected-latency constraint `L_router + Σ_s π_s·L_s + L_synth ≤ ε`
with `π_s = N_s/N`. **That formula is physically wrong for parallel specialists.**
`Σ_s π_s·L_s` is the expected *total work* under **serial** execution; it does not
model a parallel stage whose wall-clock is a **max**, not a sum. Implementing the
paper's own sketch would mis-model the architecture.

The architecturally-correct expected latency under parallel execution is

```
E_q[ max_{s ∈ active(q)} L_s ]      (mean over queries of the slowest ACTIVE specialist)
```

**This is linearisable** (I initially mis-judged it as not; corrected): with active
set `A_q` fixed from gold `expected_agents`,

```
L_q ≥ Σ_k L[s,k]·x[s,k]    ∀ s ∈ A_q        (~67 lower-bound constraints)
(1/N)·Σ_q L_q ≤ ε                            (the SLO)
```

≈ 25 aux vars + ~67 constraints — fully tractable. Under **router-induced** active
sets, `A_q(k_d)` is variable and the per-query max needs z-style/big-M gating
(harder but doable); under **gold** active sets it is clean.

**Resolution.**
- *Code option:* add `latency_model ∈ {worst_case (default), expected_max}`,
  implementing the **correct** per-query-max variant — never replacing the default.
- *Paper note:* state that the naive `Σ_s π_s·L_s` is a serial-work proxy, not the
  parallel-latency expectation; report worst-case vs expected_max as an ablation.
  This converts a hand-wave into a result.

---

## Finding 4 — The objective is an unnormalised sum of incommensurable terms

**Verified composition at an optimum** (from `milp.py`): router = 1 summand in
[0,1]; synth = 1 summand in [0,1]; specialists = up to 9 summands (one active
config per specialist), each in [0,~1]. Specialist block can reach ~9 vs ~1 for
router and ~1 for synth.

Accounting for the legitimate recall-weighting (Finding 1), the *realistic*
specialist block ≈ `Σ_s recall_s·condQ_s` ≈ 9·0.48 ≈ **4.3** (mean recall ~0.8,
condQ ~0.6) vs router F1 ~**0.85**. Ratio ≈ **5:1, specialist-dominant**.

**Consequence.** The MILP's search pressure concentrates on specialist allocation;
router and synth configs are ~1/11 of total objective mass each — effectively
free riders. This **re-introduces the very router-underweighting v3 §1.3 set out
to eliminate**, by a different mechanism (term count rather than a missing gating).
The paper argues the router must not be underweighted, then adopts an objective
that underweights it relative to the nine specialists.

**Resolution.**
- *Code option:* expose `(w_rt, w_syn, w_spec)` and/or normalise the specialist
  term to a **mean over activated specialists** instead of a sum. Mean-normalisation
  makes all three terms live in [0,1] and is the most defensible default.
- *Paper argument:* state the weighting explicitly and report sensitivity. An
  unstated weighting that emerges from how many agents are in each role is not a
  modelling decision — it's an accident. Make it a decision.

---

## Finding 4′ — Routing *precision* is under-incentivised (precise statement)

Distinct from #4 and worth stating exactly. Q³ rewards router **recall** (firing
the right specialist). It gives **no reward for precision**: a router false
positive adds a specialist that the objective never credits (the Q³ term is simply
not summed) but that costs latency and energy. Routing precision is rewarded
**only** through the router F1 term (max ~1), competing against a ~4–5 specialist
block. **Net: the optimiser is ~5× more sensitive to specialist allocation than to
routing precision; false positives are deterred by the latency constraint, not by
quality.** So precision control is indirect and weak.

**Resolution (paper argument; optional code).** Either (i) accept that latency is
the precision regulator and say so, or (ii) add an explicit per-false-positive
energy/latency charge so precision is priced. Note this depends on Finding 3's
latency model: under worst-case latency, a false-positive *slow* specialist is
penalised; under expected_max, a false positive only matters if it's the slowest
*active* one. The interaction should be stated.

---

## Finding 5 — Specialist quality excludes correctness (defended, but costs validity)

**Verified** (`factories.make_real_factories`): in real mode, specialists get
`SpecialistScorer(include_correctness=False)`; only `A_synth` gets `True`. So the
specialist Q driving the (dominant) bilinear term is `mean(faithfulness,
relevancy)` — **no correctness component**.

**Steelman (holds).** The gold `ground_truth_answer` describes the *full composed*
answer (often spanning several specialists); scoring a single specialist's partial
answer against it on correctness would systematically under-rate it. Excluding
correctness for specialists is a reasonable guard, and the value is still *measured*
and saved to parquet (only its inclusion in Q changes).

**Validity cost (real).** The objective term carrying ~5× the weight of router and
synth contains **zero correctness signal**. The MILP maximises grounded, on-topic,
*possibly legally wrong* specialist outputs. For a legal assistant whose stated
selling point is correctness under confidentiality, the dominant optimisation
signal not measuring correctness is the single biggest validity gap in the method.

**Resolution.**
- *Dataset change (highest value):* author **per-specialist** ground truth (a gold
  slice per specialist domain), so correctness can be measured fairly *and*
  included for specialists. This is the legitimacy fix; everything else is mitigation.
- *Paper argument:* until per-specialist GT exists, state plainly that specialist Q
  is a grounding/relevancy proxy and that correctness enters the system only via the
  synthesiser term.

---

## Finding 6 — Shared-embedder circularity

**Verified** (`embeddings.py`, `scorer.py`): the **same** BGE-M3 model is used for
FAISS retrieval *and* for the cosine correctness/faithfulness/relevancy scores. A
model that parrots retrieved passages scores high on faithfulness (similar to
context) and on correctness (context overlaps gold) **without being legally
correct**. Faithfulness and correctness are thus not independent measurements;
they share the retriever's geometry.

**Resolution (measurement change + caveat).** Score correctness with a *different*
embedder than the retriever, or add an LLM judge for correctness on a subset, and
report agreement. At minimum, declare the shared-embedder dependency as a known
inflation of grounding scores.

---

## Finding 7 — Cosine metrics: compressed range and regurgitation reward

Two coupled issues, both real:

1. **Compressed dynamic range.** Cosine between two same-language, same-domain legal
   texts has a high floor: a plausibly-wrong legal-sounding answer sits ~0.5–0.65
   to the ground truth; a correct one ~0.8–0.9. Usable range ≈ 0.4. The metric
   barely separates right from confidently-wrong.
2. **Regurgitation reward under oracle retrieval.** `faithfulness = max_i
   cos(answer, ctx_i)`; under oracle mode the contexts *are* the gold passages, so
   verbatim copying any passage yields faithfulness ≈ 1.0. It rewards extraction,
   not reasoning, and is near-trivially maxed.
   Also, `relevancy = cos(answer, question)` rewards topical echo, not correctness,
   yet is half the non-correctness mass.

**Resolution (paper caveat + validation).** Report these as weak proxies with
explicit limits; validate against a small human-scored subset; prefer **live
retrieval** over oracle for the headline faithfulness number so copying is not
trivially rewarded.

---

## Finding 8 — Small per-specialist sample sizes

**Verified** from `calib_clean_with_gold_text.yaml` (25 queries, 67 specialist
samples): N_s = 20, 13, 8, 8, 5, 5, **3, 3, 2**. Three specialists have ≤ 3
samples; their per-config Q — hence their slice of the allocation — is essentially
noise.

**Resolution (dataset + post-hoc).** More data for the thin specialists; and the
deepdive's **bootstrap** (resample queries, recompute Q³, re-solve, report how
often each allocation wins) to attach confidence to the frontier instead of
implying unwarranted precision. The bootstrap is pure post-processing on the
parquet — no re-measurement — and directly addresses the "noisy Q" critique a
reviewer will raise.

---

## Recommended order of operations (for the paper)

1. **State the objective weighting (4) and the latency model (3) explicitly.** These
   are the two choices that most change the optimum and are currently implicit or
   under-justified.
2. **Add the two code options** (latency_model = expected_max; objective weights /
   mean-normalised specialist term) so worst-case-vs-expected and
   weighted-vs-unweighted become **ablations with numbers**, not assertions.
3. **Fix the measurement validity (5, 6, 7)** — per-specialist GT is the big one;
   distinct scoring embedder and live-retrieval faithfulness are the next.
4. **Bootstrap the frontier (8)** as post-hoc, to report confidence.

The single most consequential item is **(5) per-specialist ground truth**: it is
the difference between "the MILP optimises measured grounding" and "the MILP
optimises measured *correctness*," which is the whole point of a legal assistant.

---

## What is NOT changed, and why

- **The worst-case latency constraint** stays as the default — it is a defended,
  correct modelling choice (v3 §9.1). The expected-latency variant is *added*, not
  substituted.
- **Q³ normalisation** is left untouched — it is correct (Finding 1 withdrawn).
- No metric is silently swapped — metric choice is the author's, and the audit
  argues for changes rather than imposing them in code.
