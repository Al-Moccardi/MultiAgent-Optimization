# MAMAP — final reframing (serial headline + serial code mode)

## Decision
Your figures and frontier data (syscap3_frontier.pdf, perf_capacity_frontier.csv with
its k in {1,3,5,9} column, x-axis "eps = L_disp + k*L_spec + L_synth") present the
SERIAL latency model. Those figures cannot be regenerated. Therefore the paper's
HEADLINE is the serial model -- matching the existing figures -- and the batched/
concurrent (max-of-one) model is the robustness check in section 10.5. To close the
paper<->code gap, the released solver now also exposes the serial mode, so the
headline frontier is reproducible from the released code.

## Paper changes (mamap_full.tex) -- now SERIAL-primary
- Title: no colon. Subtitle: "A measured, coupling-aware mixed-integer optimization
  for an on-device multi-agent legal-assistant pipeline".
- Section 6.2: primary model is the SERIAL sum, Eq. (7) = L_disp + k*L_spec + L_synth.
  "Why a sum, and the batched alternative" frames batching as the tested extreme.
- Section 6.4 + 9.1 MILP: latency constraints carry the k multiplier (k*sum spec term).
- Section 7: serial frontier with k in {1,3,5,9} curves; eps ~= 9/19/25/37 s for
  k = 1/3/5/9 restored; figure captions describe the serial k-family. NO red flags
  (the figures are CORRECT for the serial headline).
- Table 3 + chain_breakdown captions: serial k=3 framing restored.
- Section 10.5: serial is primary, batched is the tested extreme; sigma=1 serial,
  sigma=1/k batched; "batching shifts the frontier left".
- Abstract / intro Q4 / section 3 / conclusion / Limitations: serial-primary wording.
- Reproducibility note: states the headline frontier is reproduced with
  "--latency-model sequential --k-activated k"; batched model is the worst_case mode.
- ALL latency-related NEEDS-INFO flags removed (the figures now match the text).

## Code changes (MAMAP codebase) -- serial mode shipped
- part1_allocation/optimize/milp.py: added latency_model="sequential" (+k_activated).
  Constraint: L_router + k_activated*L_spec_one + L_synth <= eps, with L_spec_one>=L_s.
  Default stays "worst_case" (unchanged behaviour for existing callers).
- part1_allocation/optimize/pareto.py: latency_grid spans the serial scale when
  latency_model="sequential"; build_frontier threads latency_model/k_activated.
- part1_allocation/pipeline/run_all.py: new CLI "--latency-model sequential
  --k-activated K".
- part1_allocation/tests/test_sequential_latency.py: 4 new unit tests (arithmetic
  pinned: concurrent 5.4s; serial k=3 -> 9.4s; infeasible at concurrent budget;
  k=1 == worst_case). Full part1 suite: 42 passed.

## Reproduce the headline frontier from the released code
    python -m part1_allocation.pipeline.run_all \
        --perf-table shared/pareto/perf_table.parquet \
        --latency-model sequential --k-activated 3
  (sweep k in {1,3,5,9} for the four curves; the worst_case default gives the
   batched robustness curve of section 10.5.)

## Now mutually consistent
paper text (serial) == figures (serial k-curves) == data (k column) == code
(sequential mode reproduces them). Batched model is the robustness check in both
paper (10.5) and code (worst_case). No outstanding latency inconsistency.

## Still open (carried from prior review, unrelated to latency)
- Dynamic-layer headline numbers (F1 0.33->0.50, OOD 0.60) need bge-m3; ship the
  frozen score cache (build_score_cache.py) for offline reproduction.
- 25-profile set shares 6 queries with the 64-test set (mild optimistic bias);
  disclosed in Limitations.
