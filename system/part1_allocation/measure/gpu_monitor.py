"""
part1_allocation/measure/gpu_monitor.py
=======================================
NVIDIA GPU telemetry used by the performance sweep:

  * total used VRAM (GiB)  -> peak-memory measurement (delta across load + warmup)
  * board power draw (W)   -> energy measurement (integrated over a generate() call)

Design notes
------------
* We measure *total* used VRAM and take a **delta** (used-after-warmup minus
  used-before-load), NOT per-process memory. Per-process accounting
  (`nvidia-smi --query-compute-apps`) is **unavailable on consumer Windows GPUs in
  WDDM driver mode** -- it returns "N/A". Total-used works everywhere. The delta is
  valid because the sweep loads exactly one model at a time, so the change in used
  VRAM during that window is that model's footprint (weights + KV cache + buffers).
* We prefer **NVML** (in-process, fast, WDDM-safe) via `pynvml`; if it is not
  installed we fall back to parsing `nvidia-smi`. If neither works, every reading is
  NaN and the caller degrades gracefully (the columns stay empty, exactly as before).
* Nothing here ever raises: telemetry is best-effort.

Optional, recommended for low-overhead sampling:
    pip install nvidia-ml-py      # provides the `pynvml` module
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

# --------------------------------------------------------------------------- #
# NVML backend (preferred). Lazy, cached, never raises.
# --------------------------------------------------------------------------- #
_nvml = None          # the pynvml module once initialised; False if unavailable
_nvml_handle = None


def _try_nvml() -> bool:
    global _nvml, _nvml_handle
    if _nvml is not None:
        return _nvml is not False
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        _nvml = pynvml
        return True
    except Exception:
        _nvml = False
        return False


def _nvidia_smi(query: str) -> float:
    """Run a single-field nvidia-smi query and return the first row as float."""
    if shutil.which("nvidia-smi") is None:
        return float("nan")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL).decode().strip().splitlines()
        return float(out[0].split(",")[0].strip())
    except Exception:
        return float("nan")


# --------------------------------------------------------------------------- #
# Readings (GiB / Watts). NaN when unavailable.
# --------------------------------------------------------------------------- #
def gpu_used_mem_gb() -> float:
    """Total used VRAM on GPU 0, in GiB."""
    if _try_nvml():
        try:
            return _nvml.nvmlDeviceGetMemoryInfo(_nvml_handle).used / (1024 ** 3)
        except Exception:
            pass
    mib = _nvidia_smi("memory.used")
    return mib / 1024.0 if mib == mib else float("nan")   # MiB -> GiB


def gpu_power_w() -> float:
    """Instantaneous board power draw on GPU 0, in Watts."""
    if _try_nvml():
        try:
            return _nvml.nvmlDeviceGetPowerUsage(_nvml_handle) / 1000.0  # mW -> W
        except Exception:
            pass
    return _nvidia_smi("power.draw")


def telemetry_backend() -> str:
    if _try_nvml():
        return "nvml"
    if shutil.which("nvidia-smi") is not None:
        return "nvidia-smi"
    return "none"


# --------------------------------------------------------------------------- #
# Peak-memory sampler: tracks the maximum used VRAM over a window in a thread.
# Use as a context manager around (load + warmup); read `.peak_gb` afterwards.
# --------------------------------------------------------------------------- #
class GpuMemPeakSampler:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._peak = float("-inf")
        self._stop: threading.Event | None = None
        self._thr: threading.Thread | None = None

    def start(self) -> "GpuMemPeakSampler":
        self._peak = float("-inf")
        self._stop = threading.Event()

        def loop():
            while not self._stop.is_set():
                u = gpu_used_mem_gb()
                if u == u:                     # not NaN
                    self._peak = max(self._peak, u)
                time.sleep(self.interval_s)

        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()
        return self

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def peak_gb(self) -> float:
        return self._peak if self._peak != float("-inf") else float("nan")


# --------------------------------------------------------------------------- #
# Self-check: print, in plain terms, whether memory & energy collection will work.
# --------------------------------------------------------------------------- #
def self_check(verbose: bool = True) -> dict:
    """Probe what telemetry is actually available on THIS machine and report it."""
    backend = telemetry_backend()
    mem = gpu_used_mem_gb()
    pw = gpu_power_w()
    mem_ok = mem == mem        # not NaN
    pw_ok = pw == pw
    status = {"backend": backend, "mem_ok": mem_ok, "power_ok": pw_ok,
              "used_mem_gb": mem, "power_w": pw}
    if verbose:
        print("[self-check] GPU telemetry")
        print(f"    backend           : {backend}"
              + ("" if backend != "none" else "  (no NVML, no nvidia-smi)"))
        if mem_ok:
            print(f"    used VRAM readable : YES  ({mem * 1024:.0f} MiB now)  "
                  f"-> peak-memory (mu_k) measurement ENABLED")
        else:
            print( "    used VRAM readable : NO   "
                   "-> peak_mem_gb will be NaN (memory constraint will not bind)")
        if pw_ok:
            print(f"    board power readable: YES ({pw:.1f} W now)            "
                  f"-> energy (e_k) measurement ENABLED")
        else:
            print( "    board power readable: NO   "
                   "-> energy_j_per_tok will be NaN")
        if backend == "none":
            print("    hint: install NVML for low-overhead sampling:  pip install nvidia-ml-py")
    return status


if __name__ == "__main__":
    self_check()
