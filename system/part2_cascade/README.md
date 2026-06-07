# Part 2 — Adaptive Cascade (stub)

This part is the headline of the paper. It is intentionally a stub here: it
consumes the contract Part 1 produces and is the focus of the next iteration.

## What it consumes (no coupling beyond this)

```python
from shared.schema import ParetoBundle
bundle = ParetoBundle.load("../shared/pareto")   # written by Part 1
ladders = bundle.ladders        # per agent: quality-sorted rungs to climb
frontier = bundle.frontier      # static optima (the baselines to beat)
```

It also reads `shared/pareto/quality_table.parquet`, which contains
**per-(agent, query, config)** quality + `confidence` + `n_out_tokens`, plus
`perf_table.parquet`. Because every rung's quality/latency/energy for every
query is logged, the cascade can be **evaluated offline by replaying the table**
— no models need to be re-run except for a final confirmatory on-device check.

## Planned design (from the chat)

1. **Cascade policy**: for a query routed to agent `a`, try the cheapest rung,
   read its logged `confidence`; if below a per-agent threshold, escalate up the
   ladder. Cascade cost = (failed cheap rung) + (escalated rung), all derivable
   from the table.
2. **Criticality-aware thresholds**: profile each agent's end-to-end impact
   (downgrade-and-measure); set eager escalation for critical agents
   (dispatcher, upstream) and lazy for cheap leaves.
3. **Evaluation axes**: end-to-end quality · P95 latency · energy/query.
4. **Baselines**: always-big, always-small, random-escalate, uniform-threshold,
   the Part 1 **static optimum**, and an oracle router (upper bound).
5. **Thesis**: static MAMAP optimizes the *mean*; the cascade exploits per-query
   *difficulty variance* → the gap is the measured value of adaptivity.

## Suggested layout (to be created next)

```
part2_cascade/
├── replay.py        # offline cascade simulation over quality_table + perf_table
├── policy.py        # escalation policies (threshold, criticality-weighted, oracle)
├── criticality.py   # downgrade-and-measure end-to-end impact profiling
├── evaluate.py      # quality/latency/energy frontier across methods
└── pipeline/run_cascade.py
```
