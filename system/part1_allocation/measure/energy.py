"""
part1_allocation/measure/energy.py
==================================
Energy measurement abstraction. Energy is part of the per-config performance
sweep (hardware-specific). We report Joules/token so per-query energy can be
derived as n_tokens * energy_per_tok.

Implementations are intentionally thin wrappers around the platform tools; on a
laptop "energy" maps to battery drain, a real UX metric. Use NullEnergyMeter to
disable (energy columns become NaN and are simply not reported).

    powermetrics : macOS (Apple Silicon)   -- requires sudo
    rapl         : Intel/AMD via /sys/class/powercap/intel-rapl
    nvidia_smi   : discrete NVIDIA GPU power draw
    tegrastats   : NVIDIA Jetson
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class EnergyMeter:
    @contextmanager
    def measure(self):
        """Context manager yielding a dict that will contain {'joules': float}."""
        raise NotImplementedError


class NullEnergyMeter(EnergyMeter):
    @contextmanager
    def measure(self):
        d: dict = {"joules": float("nan")}
        yield d


class RaplEnergyMeter(EnergyMeter):
    """Reads cumulative energy from Intel/AMD RAPL counters (Linux)."""

    PKG = "/sys/class/powercap/intel-rapl:0/energy_uj"

    @contextmanager
    def measure(self):  # pragma: no cover  (hardware-specific)
        d: dict = {"joules": float("nan")}
        try:
            with open(self.PKG) as f:
                start = int(f.read())
            yield d
            with open(self.PKG) as f:
                end = int(f.read())
            d["joules"] = (end - start) / 1e6
        except Exception:
            yield d


class SamplingPowerMeter(EnergyMeter):
    """Generic fallback: integrate an instantaneous-power sampler over time.

    `power_w_fn` returns current power draw in watts (e.g. parse nvidia-smi or
    tegrastats). Energy = integral of power dt.
    """

    def __init__(self, power_w_fn, interval_s: float = 0.1):
        self.power_w_fn = power_w_fn
        self.interval_s = interval_s
        self.baseline_w = 0.0     # idle power to subtract -> marginal generation energy

    def set_baseline_w(self, w: float) -> None:
        import math
        self.baseline_w = w if (w == w and not math.isinf(w)) else 0.0  # ignore NaN/inf

    @contextmanager
    def measure(self):  # pragma: no cover
        import math
        import threading
        d: dict = {"joules": float("nan")}
        stop = threading.Event()
        samples: list[tuple[float, float]] = []

        def loop():
            while not stop.is_set():
                samples.append((time.perf_counter(), float(self.power_w_fn())))
                time.sleep(self.interval_s)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        try:
            yield d
        finally:
            stop.set()
            t.join()
            # subtract idle baseline (clamped >=0) -> marginal power; require >=2 valid
            valid = [(ts, max(0.0, p - self.baseline_w))
                     for (ts, p) in samples if not math.isnan(p)]
            if len(valid) >= 2:
                joules = 0.0
                for (t0, p0), (t1, p1) in zip(valid, valid[1:]):
                    joules += 0.5 * (p0 + p1) * (t1 - t0)
                d["joules"] = joules
            # else: leave joules = NaN (no usable power telemetry)


def make_energy_meter(backend: str) -> EnergyMeter:
    backend = (backend or "none").lower()
    if backend in ("none", ""):
        return NullEnergyMeter()
    if backend == "rapl":
        return RaplEnergyMeter()
    if backend in ("nvidia_smi", "tegrastats"):
        # Integrate NVIDIA board power (via NVML, or nvidia-smi as fallback) over
        # each generate() call. Returns NaN joules gracefully if no GPU telemetry.
        from .gpu_monitor import gpu_power_w
        return SamplingPowerMeter(gpu_power_w, interval_s=0.05)
    # powermetrics (macOS) still needs a platform-specific power_w_fn -> no-op.
    return NullEnergyMeter()
