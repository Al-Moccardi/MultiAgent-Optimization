"""
part1_allocation/inference/cuda_bootstrap.py
============================================
Make a CUDA-enabled llama.cpp wheel find its runtime DLLs on Windows.

A CUDA llama.dll depends on cublas64_12.dll / cudart64_12.dll, which the NVIDIA
driver does NOT include. The lightest way to provide them is pip:

    pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12

Those wheels drop the DLLs under site-packages/nvidia/<lib>/bin, which is NOT on
the Windows DLL search path by default. This module adds those folders (and any
standard CUDA Toolkit bin) via os.add_dll_directory BEFORE llama_cpp is imported,
so the dependency resolves. No-op on non-Windows or if nothing is found.
"""
from __future__ import annotations

import glob
import os
import site


def add_cuda_dll_dirs(verbose: bool = False) -> list[str]:
    if os.name != "nt":
        return []

    candidates: list[str] = []

    # site-packages roots (system + user)
    roots: set[str] = set()
    try:
        roots.update(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.add(site.getusersitepackages())
    except Exception:
        pass
    # also the directory tree of the running interpreter's libs
    import sys
    roots.add(os.path.join(sys.prefix, "Lib", "site-packages"))

    # 1) pip-installed nvidia-* runtime packages: nvidia/<lib>/bin/*.dll
    for sp in roots:
        candidates += glob.glob(os.path.join(sp, "nvidia", "*", "bin"))

    # 2) standard CUDA Toolkit installs
    candidates += glob.glob(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin")

    added: list[str] = []
    for d in candidates:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                added.append(d)
            except Exception:
                pass

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
        if verbose:
            print("[cuda] added DLL dirs:\n  " + "\n  ".join(added))
    elif verbose:
        print("[cuda] no CUDA runtime DLL dirs found "
              "(install: pip install nvidia-cublas-cu12 nvidia-cuda-runtime-cu12)")
    return added
