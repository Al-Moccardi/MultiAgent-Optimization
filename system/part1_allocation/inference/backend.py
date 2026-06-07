"""
part1_allocation/inference/backend.py
=====================================
Inference abstraction so the rest of the pipeline never imports llama.cpp
directly. Two implementations:

  * LlamaCppBackend : real, via `llama-cpp-python`. Returns token-level logprobs
    so we can compute the confidence signal the cascade (Part 2) needs.
  * MockBackend     : deterministic, no model. Lets you run the whole pipeline
    (tables -> MILP -> Pareto -> shared/) without any GGUF files.

Standardize on GGUF + llama.cpp so the SAME artifact runs on CPU / Metal / CUDA,
which is what makes "quality measured once" valid across hardware classes.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass


@dataclass
class GenResult:
    text: str
    n_out_tokens: int
    ttft_s: float
    total_s: float
    # mean per-token logprob of the generated answer (None if unavailable)
    mean_logprob: float | None = None

    @property
    def throughput_tok_s(self) -> float:
        gen_s = max(self.total_s - self.ttft_s, 1e-6)
        return self.n_out_tokens / gen_s

    @property
    def confidence(self) -> float | None:
        """Map mean logprob -> (0,1] via perplexity. confidence = 1/perplexity."""
        if self.mean_logprob is None:
            return None
        return float(math.exp(self.mean_logprob))  # exp(mean logprob) in (0,1]


class InferenceBackend:
    def generate(self, prompt: str, *, max_tokens: int = 512,
                 context: int | None = None) -> GenResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LlamaCppBackend(InferenceBackend):
    """Real backend. Requires `pip install llama-cpp-python` built for your HW.

    Build hints (pick the one matching the device class you are profiling):
        CPU         : pip install llama-cpp-python
        Apple Metal : CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
        CUDA        : CMAKE_ARGS="-DGGML_CUDA=on"  pip install llama-cpp-python
    """

    def __init__(self, gguf_path: str, n_ctx: int = 8192, n_gpu_layers: int = -1,
                 seed: int = 0, want_logprobs: bool = True, **kwargs):
        try:
            from .cuda_bootstrap import add_cuda_dll_dirs
            add_cuda_dll_dirs()
        except Exception:
            pass
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install it built for your hardware (see class docstring).") from e
        # logprobs (for the confidence signal) require logits_all=True in llama.cpp.
        self._want_logprobs = want_logprobs
        self._llm = Llama(model_path=gguf_path, n_ctx=n_ctx,
                          n_gpu_layers=n_gpu_layers, seed=seed,
                          logits_all=want_logprobs, verbose=False, **kwargs)

    def generate(self, prompt: str, *, max_tokens: int = 512,
                 context: int | None = None) -> GenResult:
        t0 = time.perf_counter()

        # STREAMING decode: timestamp the first emitted token for a REAL TTFT, and
        # measure decode throughput over the generation phase only (excludes prompt
        # prefill). Greedy (temperature=0) for determinism -> hardware-invariant.
        kw = dict(prompt=prompt, max_tokens=max_tokens, temperature=0.0, stream=True)
        if self._want_logprobs:
            kw["logprobs"] = 1
        try:
            stream = self._llm.create_completion(**kw)
        except (ValueError, TypeError):
            kw.pop("logprobs", None)
            self._want_logprobs = False
            stream = self._llm.create_completion(**kw)

        text_parts: list[str] = []
        token_logprobs: list[float] = []
        n_out = 0
        t_first: float | None = None
        try:
            for chunk in stream:
                choice = chunk["choices"][0]
                piece = choice.get("text", "")
                if piece and t_first is None:
                    t_first = time.perf_counter()      # real time-to-first-token
                if piece:
                    text_parts.append(piece)
                    n_out += 1
                lp = choice.get("logprobs") or {}
                for x in (lp.get("token_logprobs") or []):
                    if x is not None:
                        token_logprobs.append(x)
        except Exception:
            pass                                       # use whatever we collected

        total_s = time.perf_counter() - t0
        text = "".join(text_parts)
        ttft_s = (t_first - t0) if t_first is not None else total_s
        mean_lp = (sum(token_logprobs) / len(token_logprobs)) if token_logprobs else None

        # GenResult.throughput_tok_s = n_out / (total_s - ttft_s): with the REAL
        # ttft above this is now a genuine decode rate, not the old coarse proxy.
        return GenResult(text=text, n_out_tokens=int(n_out), ttft_s=float(ttft_s),
                         total_s=float(total_s), mean_logprob=mean_lp)

    def close(self) -> None:  # pragma: no cover
        del self._llm


class MockBackend(InferenceBackend):
    """Deterministic stand-in. Output length and 'confidence' are a stable
    function of (prompt, config tag) so dry runs are reproducible."""

    def __init__(self, config_tag: str = "mock", quality_hint: float = 0.5,
                 params_b: float | None = None, quant: str = "Q4_K_M",
                 context: int = 8192):
        self.config_tag = config_tag
        self.quality_hint = quality_hint  # bigger models -> set higher in factory
        # Synthetic peak memory: weights (quant-dependent) + KV cache + overhead.
        p = params_b if params_b is not None else 1.0
        bytes_per_param = 1.05 if (quant.startswith("Q8") or quant.upper() in ("F16", "FP16")) else 0.55
        weights_gb = p * bytes_per_param
        kv_gb = (context / 8192.0) * p * 0.10
        self.peak_mem_gb = round(weights_gb + kv_gb + 0.4, 3)

    def _seed(self, prompt: str) -> int:
        h = hashlib.sha256((self.config_tag + "|" + prompt).encode()).hexdigest()
        return int(h[:8], 16)

    def generate(self, prompt: str, *, max_tokens: int = 512,
                 context: int | None = None) -> GenResult:
        s = self._seed(prompt)
        n_out = 120 + (s % 400)                    # 120..519 tokens
        # better configs (quality_hint high) -> higher mean logprob (closer to 0)
        mean_lp = -(0.05 + (1.0 - self.quality_hint) * 0.6) - ((s % 7) * 0.01)
        ttft = 0.15 + (1.0 - self.quality_hint) * 0.05
        tput = 30.0 + self.quality_hint * 40.0
        total = ttft + n_out / tput
        text = f"[mock:{self.config_tag}] answer to: {prompt[:48]}..."
        return GenResult(text=text, n_out_tokens=n_out, ttft_s=ttft,
                         total_s=total, mean_logprob=mean_lp)
