# MAMAP-Edge

Two-part codebase for the paper *Multi-Agent Model Allocation + adaptive cascade
for resource-constrained on-device deployment* (TETCI special issue on Efficient
Deep Learning in Resource-Constrained Environments).

```
mamap-edge/
├── shared/                 # the CONTRACT between the two parts
│   └── schema.py           #   dataclasses + (de)serialization of shared/pareto/
├── part1_allocation/       # PART 1 — MAMAP MILP + empirical Pareto frontier   <-- THIS RELEASE
├── part2_cascade/          # PART 2 — adaptive cascade (stub; consumes shared/pareto/)
└── shared/pareto/          # produced by Part 1, consumed by Part 2
```

The interface between the parts is the directory **`shared/pareto/`**. Part 1
writes it; Part 2 reads it. Nothing else couples them.

---

## The idea in one paragraph

Part 1 measures every candidate config **on the real device** (no third-party
proxies, no params-as-capability heuristic) and solves a MILP whose objective is
**measured end-to-end quality**. It emits a capacity(=quality)–latency Pareto
front plus, per agent, a **ladder** of configs. Part 2 then runs an *adaptive
cascade* that climbs those ladders per query, beating the best static allocation
by exploiting per-query difficulty variance.

Two data types, treated differently (this is the key to generalizing across
hardware cheaply):

| table | keyed by | cost | reuse |
|---|---|---|---|
| **quality** | (agent, query, config) | expensive | measured **once** — hardware-invariant |
| **performance** | (hardware, config) | cheap | re-measured **per device** |

Per-query latency / energy are **derived**, never separately measured:
`L = ttft + n_tokens / throughput`, `E = n_tokens · energy_per_tok`.

---

## Quickstart (no models needed)

```bash
pip install -r requirements.txt          # core: pulp, pyyaml, pandas, pyarrow
export PYTHONPATH=.
python -m part1_allocation.tests.test_milp          # MILP sanity tests
python -m part1_allocation.pipeline.run_all --mode mock
```

This runs the entire pipeline with a deterministic `MockBackend`/`MockScorer`
(synthetic quality, latency and memory) and writes a real Pareto front to
`shared/pareto/`. Use it to validate the plumbing and inspect the artifacts.

---

## Real run (llama.cpp + RAGAS, on your device)

### 1. Install the heavy deps

```bash
pip install -r requirements-real.txt
# llama-cpp-python must be built for the device class you are profiling:
#   CPU         : pip install llama-cpp-python
#   Apple Metal : CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
#   CUDA        : CMAKE_ARGS="-DLLAMA_CUDA=on"  pip install llama-cpp-python
```

### 2. Fill in the config

* **Device — auto-detected.** You do *not* need to write `device.yaml`. Run with
  `--device auto` and the device is probed: `hw_class`, memory budget (M), name,
  and energy backend come from the machine. The only things you supply are the
  ones that are *requirements, not device facts*:
  `--latency-sla <seconds>` (T°) and `--latency-metric mean|p95`. Optionally tune
  the memory headroom with `--memory-fraction 0.7` or pin it with
  `--memory-budget-gb 12`. Inspect what was detected with:

  ```bash
  python -m part1_allocation.measure.device_probe
  ```

  (You can still hand-write a `device.yaml` and pass `--device path` if you prefer
  full control.)
* `part1_allocation/config/catalog.yaml` — set `gguf_path` for each
  (model, quant). You need ~5 models × 2 quant ≈ **10 GGUF files** — context
  length is a llama.cpp *runtime* parameter (`n_ctx`), not a separate file, so the
  same GGUF serves all context rows. **Vet Italian competence before including.**
* `part1_allocation/config/agents.yaml` — already contains the 9 specialists +
  the dispatcher; tune each agent's `c_min`.
* `part1_allocation/data/testset.example.jsonl` — replace with your labeled set
  (~20–30 queries per agent). Fields: `query_id, agent, question, contexts[],
  ground_truth`. Pre-fill `contexts` with your RAG-retrieved chunks (then Part 1
  needs no live retriever).
* `part1_allocation/data/calib_clean.yaml` — the **routing gold set**. Each query
  carries `expected_agents` (multi-label), `difficulty`, `risk_level`, `category`.
  It is used automatically (`--calib`, on by default) to evaluate the
  **dispatcher**: the dispatcher's *quality is multi-label routing F1*, not RAGAS.
  Add `--calib-fanout` to also push each query to its expected specialist agents
  (carrying the difficulty/risk labels). The labels propagate into
  `quality_table.parquet` (columns `difficulty, risk_level, category,
  expected_agents`) so you get quality-by-difficulty breakdowns and the signals
  Part 2's cascade + criticality policy need. After the sweep the runner prints a
  routing F1 summary; `measure.routing_report.routing_report(df)` gives the full
  per-config / per-difficulty table.

### 3. Wire the device-specific hooks (see `factories.py` / `run_all.py`)

* **RAGAS judge** — pass a judge LLM + embeddings into `make_real_factories`.
  For fully offline scoring, point the judge at a strong local model.
* **Legal correctness** — RAGAS measures grounding, **not** legal correctness.
  Supply a `correctness_fn(sample, answer)->[0,1]` (expert-checked subset or a
  correctness judge); it is blended into the objective via `aggregate(...)`.
* **Peak memory probe** — set `peak_mem_probe` in `run_all.py` for real mode
  (e.g. `torch.cuda.max_memory_allocated`, Metal allocation, or an RSS delta via
  `psutil`). Without it, `peak_mem_gb` is NaN and the MILP cannot bind memory.
* **Energy** — set `energy_backend` in `device.yaml` (`rapl`/`powermetrics`/
  `nvidia_smi`/`tegrastats`) and wire a power sampler in `measure/energy.py`.

### 4. Run, and reuse quality across devices

```bash
# Device A: full run (measures quality once + performance for A)
python -m part1_allocation.pipeline.run_all --mode real --device config/device_A.yaml

# Device B: reuse the (hardware-invariant) quality table, only re-measure perf
python -m part1_allocation.pipeline.run_all --mode real --device config/device_B.yaml \
    --reuse-quality shared/pareto/quality_table.parquet
```

The reuse flag is the whole point of the data split: generalizing to a new
laptop only re-pays the **cheap** performance sweep.

> Determinism note: use greedy decoding (already set) and identical GGUF
> artifacts across devices so outputs — and thus quality — match. Spot-check
> cross-device output agreement to justify "quality measured once".

---

## What Part 1 produces (the contract)

`shared/pareto/`:

| file | contents |
|---|---|
| `frontier.json` | non-dominated static allocations: `{allocation, loaded, total_quality, max_latency_s, per_agent_latency_s, …}` |
| `ladders.json`  | per agent: quality-sorted rungs `{config_id, quality, latency_s, energy_j, peak_mem_gb}` — what the cascade climbs |
| `configs.json`  | resolved config metadata |
| `manifest.json` | provenance (device, hardware, counts, timestamp) |
| `quality_table.parquet` | full per-(agent,query,config) records incl. **confidence** → lets Part 2 replay the cascade **offline** |
| `perf_table.parquet`    | per-(hardware,config) μ, ttft, throughput, energy/tok |

Load it from Python:

```python
from shared.schema import ParetoBundle
bundle = ParetoBundle.load("shared/pareto")
best_static = max(bundle.frontier, key=lambda s: s.total_quality)
```

---

## Troubleshooting the llama.cpp wheel (Windows)

- **`Could not find module 'llama.dll' (or one of its dependencies)`** after installing
  the official CUDA wheel: the wheel's CUDA runtime DLLs aren't on your system. Use a
  self-contained community CUDA wheel (bundles CUDA; no Toolkit needed), or install the
  matching CUDA Toolkit + VC++ redistributable.
- **`Windows Error 0xc000001d`** (illegal instruction) when *loading* a model: the CPU
  wheel was built for SIMD your CPU lacks (often AVX-512, disabled on many consumer Intel
  laptops). Fixes: run on GPU with a self-contained CUDA wheel, or use a CPU wheel built
  generic (`GGML_NATIVE=OFF`) / AVX2-only.
- **Recommended for an NVIDIA laptop:** a self-contained CUDA wheel matching your Python
  version and GPU arch (Ada/sm_89 for a 40-series). cp312 wheels covering 10-50 series are
  easy to find; on Python 3.11 either locate a cp311 wheel or make a Python 3.12 venv.

A single model that fails to load (OOM, incompatible) is now skipped with a warning rather
than crashing the sweep; if *all* fail you get a clear message pointing here.

## Scaling to a large catalog (SLM zoo)

The MILP scales fine to hundreds of configs (a 252-config × 10-agent solve is
~0.6 s; the full Pareto sweep ~2 s). To build a big catalog without hand-writing
entries, use the generator over the curated registry
`part1_allocation/tools/slm_zoo.yaml`:

```bash
# resolve real GGUF filenames from HuggingFace and (optionally) download them
python -m part1_allocation.tools.gen_catalog --quants Q4_K_M --contexts 4096 \
    --max-params 7.2 --download --out part1_allocation/config/catalog.zoo.yaml

# then run the router over the whole zoo (cheap: routing quality needs no judge)
python -m part1_allocation.pipeline.run_all --mode real --device auto \
    --catalog part1_allocation/config/catalog.zoo.yaml --only-agents A_dispatcher
```

`--max-params` drops models too big for your device. Add/remove models by editing
the registry. A bad/non-Italian model simply scores low routing F1 and the MILP
won't pick it — the measurement is self-cleaning.

**Quality dedup.** Quality is measured once per `(model, quant)` and reused across
context-length variants (the output is context-invariant), cutting the expensive
sweep ~C-fold for C context lengths. Disable with `--no-quality-dedup`.



* **MILP / model sharing** — `optimize/milp.py` (x/y split, eq. 4–8).
* **ε-constraint Pareto** — `optimize/pareto.py` (eq. 10–11).
* **Proxy-free objective** — objective coefficient is measured `Q_{a,k}`, not
  params; see `measure/quality.py` + `optimize/derive.py`.
* **Two-scale adaptivity** — Part 1 = adapt allocation to *device*; Part 2 =
  adapt config to *query*.
