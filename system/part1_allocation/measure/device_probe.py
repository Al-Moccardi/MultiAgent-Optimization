"""
part1_allocation/measure/device_probe.py
========================================
Auto-detect the device properties that ARE contained on the device, so a
non-specialist user does not have to fill in device.yaml by hand.

Detected automatically:
  * hw_class        -> UNIFIED (Apple Silicon) | DISCRETE_GPU (NVIDIA) | CPU_ONLY
  * memory_budget   -> total memory of the relevant class * headroom fraction
  * name            -> hostname + chip / GPU name
  * energy_backend  -> best-effort (powermetrics / nvidia_smi / rapl / none)

NOT detected (these are requirements / methodology, not device facts):
  * latency_sla_s   (T°)  -- how reactive YOU need the assistant to be
  * latency_metric  (mean | p95)

Memory headroom: an LLM assistant shares the laptop with the OS and the user's
other apps (browser, Word, ...), so we never claim 100% of memory. We detect the
TOTAL and apply a fraction (configurable); the result is fully transparent.

Dependency-light: uses psutil if present, otherwise platform-specific fallbacks.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

from shared.schema import DeviceSpec

DEFAULT_FRACTION_SHARED = 0.70   # UNIFIED / CPU_ONLY: leave room for OS + apps
DEFAULT_FRACTION_VRAM = 0.90     # DISCRETE_GPU: dedicated, less contention


# --------------------------------------------------------------------------- #
# Low-level probes (each returns None on failure; never raises)
# --------------------------------------------------------------------------- #
def _total_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                          stderr=subprocess.DEVNULL).decode().strip()
            return int(out) / (1024 ** 3)
        if system == "Linux":
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024 ** 3)
    except Exception:
        return None
    return None


def _nvidia_gpu() -> tuple[float, float, str] | None:
    """Return (vram_total_gib, vram_free_gib, gpu_name) for GPU 0, else None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free,name",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        total_mib, free_mib, name = out[0].split(",", 2)
        return (float(total_mib) / 1024.0, float(free_mib) / 1024.0, name.strip())
    except Exception:
        return None


def _apple_chip() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return platform.processor() or "Apple-Silicon"


def _has_rapl() -> bool:
    return (os.path.isdir("/sys/class/powercap/intel-rapl:0")
            or os.path.isdir("/sys/class/powercap/intel-rapl"))


def _is_jetson() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "jetson" in f.read().lower()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_device(latency_sla_s: float = 8.0,
                  latency_metric: str = "p95",
                  memory_fraction: float | None = None,
                  memory_budget_gb: float | None = None,
                  name: str | None = None) -> DeviceSpec:
    """Detect a DeviceSpec. Override any detected field via the kwargs.

    latency_sla_s / latency_metric are NOT detectable -> pass your requirement.
    memory_budget_gb, if given, overrides the auto memory budget entirely.
    """
    system = platform.system()
    machine = platform.machine().lower()
    gpu = _nvidia_gpu()

    # --- DISCRETE_GPU (NVIDIA present) ---
    if gpu is not None:
        vram_total, vram_free, gpu_name = gpu
        frac = memory_fraction if memory_fraction is not None else DEFAULT_FRACTION_VRAM
        # Budget against ACTUALLY-FREE VRAM (the desktop/OS already uses some), with
        # headroom. Falls back to total*frac if the free read looks bogus.
        base = vram_free if (vram_free and vram_free > 0.5) else vram_total
        budget = memory_budget_gb if memory_budget_gb is not None else round(base * frac, 2)
        return DeviceSpec(
            name=name or f"{platform.node()}__{gpu_name}",
            hw_class="DISCRETE_GPU", memory_budget_gb=budget,
            latency_sla_s=latency_sla_s, latency_metric=latency_metric,
            energy_backend="tegrastats" if _is_jetson() else "nvidia_smi",
        )

    # --- UNIFIED (Apple Silicon) ---
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        total = _total_ram_gb() or 16.0
        frac = memory_fraction if memory_fraction is not None else DEFAULT_FRACTION_SHARED
        budget = memory_budget_gb if memory_budget_gb is not None else round(total * frac, 2)
        return DeviceSpec(
            name=name or f"{platform.node()}__{_apple_chip()}",
            hw_class="UNIFIED", memory_budget_gb=budget,
            latency_sla_s=latency_sla_s, latency_metric=latency_metric,
            energy_backend="powermetrics",   # needs sudo; meter falls back to none
        )

    # --- CPU_ONLY (everything else) ---
    total = _total_ram_gb() or 8.0
    frac = memory_fraction if memory_fraction is not None else DEFAULT_FRACTION_SHARED
    budget = memory_budget_gb if memory_budget_gb is not None else round(total * frac, 2)
    energy = "rapl" if (system == "Linux" and _has_rapl()) else "none"
    return DeviceSpec(
        name=name or f"{platform.node()}__cpu",
        hw_class="CPU_ONLY", memory_budget_gb=budget,
        latency_sla_s=latency_sla_s, latency_metric=latency_metric,
        energy_backend=energy,
    )


def describe(dev: DeviceSpec) -> str:
    return (f"detected device:\n"
            f"  name           : {dev.name}\n"
            f"  hw_class       : {dev.hw_class}\n"
            f"  memory_budget  : {dev.memory_budget_gb} GB  (auto; M in the MILP)\n"
            f"  energy_backend : {dev.energy_backend}\n"
            f"  latency_sla_s  : {dev.latency_sla_s} s   (NOT detected -- your requirement)\n"
            f"  latency_metric : {dev.latency_metric}")


if __name__ == "__main__":
    print(describe(detect_device()))
