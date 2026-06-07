"""
part1_allocation/measure/performance.py
=======================================
Per-hardware performance sweep (CHEAP, repeat per device).

For each config on the current device we measure peak memory, TTFT, throughput
and energy/token by running a fixed warmup workload. Outputs PerfRecords.

Peak memory: prefer reading the backend/runtime's reported footprint. As a
portable fallback we expose a hook `peak_mem_probe` you wire to your platform
(e.g. metal allocations, torch.cuda.max_memory_allocated, or RSS delta).
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

from shared.schema import ConfigSpec, PerfRecord
from part1_allocation.inference.backend import InferenceBackend
from part1_allocation.measure.energy import EnergyMeter, NullEnergyMeter

# A backend factory: config -> InferenceBackend
BackendFactory = Callable[[ConfigSpec], InferenceBackend]
# Optional peak-memory probe: backend -> GB (best wired per platform)
PeakMemProbe = Callable[[InferenceBackend], float]

WARMUP_PROMPTS = [
    "Riassumi in breve la disciplina della successione legittima.",
    "Quali sono i presupposti della separazione consensuale?",
    "Spiega il regime patrimoniale della comunione dei beni.",
]


def measure_performance(configs: dict[str, ConfigSpec],
                        hardware: str,
                        backend_factory: BackendFactory,
                        energy_meter: EnergyMeter | None = None,
                        peak_mem_probe: PeakMemProbe | None = None,
                        repeats: int = 3,
                        max_tokens: int = 256) -> list[PerfRecord]:
    energy_meter = energy_meter or NullEnergyMeter()
    records: list[PerfRecord] = []

    # GPU memory is measured as a DELTA of total used VRAM across (load + warmup).
    # gpu_used_mem_gb() returns NaN on non-NVIDIA / no-telemetry boxes, in which case
    # we fall back to the caller-supplied peak_mem_probe (e.g. the mock probe).
    import gc
    from .gpu_monitor import gpu_used_mem_gb, gpu_power_w, GpuMemPeakSampler

    def _settle_vram(timeout_s: float = 3.0, eps: float = 0.05) -> float:
        """Wait until used VRAM stops dropping (model freed), return the stable value.
        Replaces a fixed sleep; robust to slow CUDA teardown (e.g. Windows WDDM)."""
        prev = gpu_used_mem_gb()
        t_end = time.perf_counter() + timeout_s
        while time.perf_counter() < t_end:
            time.sleep(0.15)
            cur = gpu_used_mem_gb()
            if not (cur == cur):            # NaN -> no telemetry, nothing to wait for
                return cur
            if abs(cur - prev) <= eps:
                return cur
            prev = cur
        return prev

    for cid, spec in configs.items():
        base_mem = gpu_used_mem_gb()          # used VRAM BEFORE this model loads
        sampler = GpuMemPeakSampler().start()  # track peak used VRAM during load+warmup
        try:
            backend = backend_factory(spec)
        except Exception as e:
            sampler.stop()
            print(f"[skip] perf '{cid}': model failed to load ({type(e).__name__}: {e})")
            continue

        # Idle power AFTER load, BEFORE generating: the marginal generation energy is
        # (power_during_gen - idle_power) integrated over time, so we subtract this.
        idle_w = gpu_power_w()
        energy_meter.set_baseline_w(idle_w) if hasattr(energy_meter, "set_baseline_w") else None

        ttfts, tputs, epts, peak = [], [], [], float("nan")
        try:
            for _ in range(repeats):
                for prompt in WARMUP_PROMPTS:
                    with energy_meter.measure() as e:
                        res = backend.generate(prompt, max_tokens=max_tokens,
                                               context=spec.context)
                    ttfts.append(res.ttft_s)
                    tputs.append(res.throughput_tok_s)
                    if res.n_out_tokens > 0 and e.get("joules") == e.get("joules"):  # not NaN
                        epts.append(e["joules"] / res.n_out_tokens)
            sampler.stop()
            peak_used = sampler.peak_gb
            if peak_used == peak_used and base_mem == base_mem:   # both not NaN
                peak = max(0.0, peak_used - base_mem)
            elif peak_mem_probe is not None:                      # fallback (e.g. mock)
                peak = peak_mem_probe(backend)
        finally:
            sampler.stop()
            backend.close()
            gc.collect()
            _settle_vram()        # poll until this model's VRAM is actually freed

        records.append(PerfRecord(
            hardware=hardware, config_id=cid,
            peak_mem_gb=peak,
            ttft_s=statistics.median(ttfts) if ttfts else float("nan"),
            throughput_tok_s=statistics.median(tputs) if tputs else float("nan"),
            energy_j_per_tok=statistics.median(epts) if epts else float("nan"),
        ))
    return records
