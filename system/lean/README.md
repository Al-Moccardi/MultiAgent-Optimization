# mamap_repo / lean

Quality-aware MAMAP — shrunken, single-objective, paper-ready.

This subdirectory sits **alongside** the colleague's `part1_static_allocation/`,
`part2_quality_aware/`, `part3_bilinear_gated/` and is intentionally narrower:

- **One model family.** Qwen2.5 in three sizes (0.5B, 1.5B, 3B), three
  quantisations (Q3_K_M, Q5_K_M, Q8_0), three context windows (2048, 4096,
  8192). `3 × 3 × 3 = 27` raw triples; ≈ 15–20 feasible at the 6.99 GB budget.
- **One objective.** Maximise pipeline quality $Q$ — dispatcher F1 plus
  averaged specialist quality plus synthesiser quality — under hard memory
  $M$ and hard latency SLA $T^\circ$. **No scalarisation weights**.
- **Concurrent latency.** Specialists sharing a $(m, q)$ group execute in a
  single batched pass: $L_\text{total} = L_d + \max_g L_{s, g} + L_y$. One
  auxiliary $\Lambda$ linearises the max.
- **Naive baselines** (largest-fits, per-role-best, uniform, random-feasible)
  for honest comparison.
- **MultiHop-RAG benchmark** with routing.
- **Reproducibility:** every result's `meta.json` carries the catalog
  SHA-256, the dataset version, the master seed, and library versions.

The colleague's parts/1–3 are kept byte-identical for reproducibility of their
claims; this lean version imports `shared/faiss_code/scorer.py` and uses
`shared/data/perf_table.parquet` directly.

## Layout

```
lean/
├── catalog/
│   ├── catalog.yaml          # Qwen2.5 × 3 sizes × 3 quants × 3 contexts
│   ├── build_catalog.py      # subset shared/perf_table.parquet + analytical w_g, κ_k → catalog.json + sidecar
│   └── catalog.json          # produced by build_catalog (committed for reproducibility)
├── multihop_rag/             # L5: HF dataset → per-(role, config, domain) quality.parquet
├── src/
│   ├── types.py              # frozen Pydantic models
│   ├── instance.py           # YAML + catalog.json → Instance, with per-role eligibility + SLA pre-filter
│   ├── latency.py            # per-group concurrent Λ encoding
│   ├── milp.py               # HiGHS: max Q s.t. memory + SLA
│   ├── baselines.py          # the four naive baselines
│   └── experiment.py         # runner emitting meta.json
├── experiments/              # YAMLs
├── tests/                    # pytest
└── paper/whitepaper.md       # algorithm box + results
```

## Quickstart

```bash
cd mamap_repo/lean
pip install -r requirements.txt

# Build the catalog from the colleague's measured perf table (no GPU, no network).
python -m catalog.build_catalog

# Validate + run a small experiment (after L2–L4 land).
python -m src.experiment experiments/lean_8gb.yaml
```

## Status

Active development. See the plan file
`~/.claude/plans/according-to-mamap-formulazione-md-vectorized-muffin.md`
(top section) for the milestone roadmap.
