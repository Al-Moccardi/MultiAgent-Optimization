"""Frozen pydantic models — single source of truth for lean MAMAP data shapes.

Mirrors the parent project (`c:\\Users\\mfoni\\Desktop\\MAMAP\\src\\mamap\\types.py`)
but is **role-aware** instead of agent-aware: dispatcher / specialist / synth
replace the generic agent. Decoupled weights + KV memory model is preserved.

Formulation (see `paper/whitepaper.md` once it lands):

    max  Q(x) = F_d(c_d) + (1/k) Σ_δ Q_s(c_s, δ) + Q_y(c_y)
    s.t. Σ_g w_g z_g + Σ_{r,c} κ_c x_{r,c}  ≤  M             (memory)
         L_d(c_d) + Λ + L_y(c_y)             ≤  T°            (concurrent latency)
         Λ ≥ L_c · x_{s,c}    for every c eligible for s        (per-group concurrent)
         Σ_c x_{r,c} = 1                     for every role r   (assignment)
         x_{r,c} ≤ z_{g(c)}                                    (load-use)
         x, z ∈ {0,1};  Λ ≥ 0
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

_Frozen = ConfigDict(frozen=True, extra="forbid")
_GB = 1024**3


class Role(StrEnum):
    """The three pipeline roles. Aggregates the colleague's 10-agent pipeline.

    The colleague's `mamap_repo/shared/data/agents.yaml` defines 10 agents
    (1 dispatcher, 9 domain-specialist + 1 synth) but every formulation in
    parts/1–3 collapses the 9 specialists into a single shared-specialist
    role because they all run the same loaded model and share its memory.
    """

    DISPATCHER = "d"
    SPECIALIST = "s"
    SYNTHESIZER = "y"


class ModelQuantGroup(BaseModel):
    """A model–quant group $g = (m, q) \\in \\mathcal{G}$ — weights shared at this level.

    Static / architectural fields (from each model's HF `config.json`). The
    `weight_gb` field is the GGUF file size on the colleague's measured
    hardware (a function of `params` and `quant`), not an analytical
    proxy — it comes from the perf_table at build time.
    """

    model_config = _Frozen

    model_id: str = Field(description="HuggingFace hfId, e.g. Qwen/Qwen2.5-3B-Instruct")
    quant: str = Field(description="quantisation label, e.g. Q5_K_M")
    params: int = Field(gt=0, description="$p_m$ — absolute parameter count")
    weight_gb: float = Field(gt=0, description="$w_g$ — weights memory in GB")
    c_max: int = Field(gt=0, description="$c^{max}_m$ — max supported context")
    n_layers: int = Field(gt=0, description="transformer layers (for KV closed form)")
    n_kv_heads: int = Field(gt=0, description="GQA/MQA key/value heads")
    head_dim: int = Field(gt=0, description="$d$ — head dimension")
    kv_dtype_bytes: float = Field(
        default=2.0, gt=0, description="$b_{kv}$ — KV element size (FP16 → 2)"
    )
    perf_prefix: str | None = Field(
        default=None,
        description=(
            "explicit stable id matching the colleague's perf/quality tables, "
            "e.g. 'qwen2.5-3b'. If set, `Config.config_id` uses "
            "`{perf_prefix}__{quant}__c{ctx}`. Falls back to a derived form."
        ),
    )

    @property
    def key(self) -> tuple[str, str]:
        return (self.model_id, self.quant)

    def kv_bytes(self, context_length: int) -> float:
        """$\\kappa_k = 2\\,L\\,n_{kv}\\,d\\,c\\,b_{kv}$ bytes."""
        return (
            2.0
            * self.n_layers
            * self.n_kv_heads
            * self.head_dim
            * context_length
            * self.kv_dtype_bytes
        )

    def kv_gb(self, context_length: int) -> float:
        return self.kv_bytes(context_length) / _GB


class Config(BaseModel):
    """One catalog row $k = (m_k, q_k, c_k) \\in \\mathcal{K}$.

    Measured fields (`ttft_s`, `throughput_tps`, `energy_j_per_tok`) come from
    the colleague's `perf_table.parquet` for the specific hardware row.
    """

    model_config = _Frozen

    group: ModelQuantGroup
    context_length: int = Field(gt=0)
    ttft_s: float = Field(gt=0, description="$\\ell_k$ — time-to-first-token (s)")
    throughput_tps: float = Field(gt=0, description="$\\tau_k$ — output tok/s")
    energy_j_per_tok: float = Field(
        gt=0, description="measured per-token energy (J), NVML-integrated"
    )

    @property
    def model_id(self) -> str:
        return self.group.model_id

    @property
    def quant(self) -> str:
        return self.group.quant

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.group.model_id, self.group.quant, self.context_length)

    @property
    def kv_gb(self) -> float:
        return self.group.kv_gb(self.context_length)

    @property
    def config_id(self) -> str:
        """Stable id used in the perf / quality tables.

        Uses ``group.perf_prefix`` if set (colleague's convention, e.g.
        ``qwen2.5-3b``). Otherwise derives from ``model_id`` for back-compat
        with synthetic catalogs that don't carry a perf_prefix.
        """
        if self.group.perf_prefix is not None:
            return f"{self.group.perf_prefix}__{self.quant}__c{self.context_length}"
        prefix = self.model_id.split("/")[-1].lower().replace("-instruct", "")
        return f"{prefix}__{self.quant}__c{self.context_length}"

    def latency(self, n_gen: int) -> float:
        """$L = \\ell + n_{gen}/\\tau$ — total wall time for one call producing `n_gen` tokens."""
        return self.ttft_s + n_gen / self.throughput_tps

    def energy(self, n_gen: int) -> float:
        """Energy per call ≈ per-token-J × n_gen (TTFT energy folded in by the measurement)."""
        return self.energy_j_per_tok * n_gen

    @model_validator(mode="after")
    def _ctx_within_cmax(self) -> Config:
        if self.context_length > self.group.c_max:
            raise ValueError(
                f"Config ctx {self.context_length} exceeds group c_max {self.group.c_max}"
            )
        return self


class Catalog(BaseModel):
    """Set $\\mathcal{K}$ of candidate configurations.

    Built once by `lean/catalog/build_catalog.py` and persisted as
    `catalog/catalog.json` with a SHA-256 sidecar (reproducibility anchor).
    Configs sharing the same `(model_id, quant)` reference a single
    `ModelQuantGroup` instance on load.
    """

    model_config = _Frozen

    configs: tuple[Config, ...]

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> Config:
        return self.configs[idx]

    @model_validator(mode="after")
    def _no_duplicate_keys(self) -> Catalog:
        keys = [c.key for c in self.configs]
        if len(set(keys)) != len(keys):
            raise ValueError("Catalog contains duplicate (model, quant, ctx) keys")
        return self

    @property
    def groups(self) -> tuple[ModelQuantGroup, ...]:
        """Unique `ModelQuantGroup` objects, in stable order."""
        seen: dict[tuple[str, str], ModelQuantGroup] = {}
        for c in self.configs:
            seen.setdefault(c.group.key, c.group)
        return tuple(seen.values())


class Instance(BaseModel):
    """A complete lean MAMAP problem instance.

    $T^\\circ$ is the **hard** end-to-end latency SLA (concurrent model
    $L_d + \\Lambda + L_y$). $n_d$ / $n_s$ / $n_y$ are the assumed generated-
    token budgets per role used to derive $L$ and $E$ from the measured
    `ttft_s` / `throughput_tps`. `domains` carries the per-query active domain
    set for the specialist (see `Q_s` averaging in the objective).
    """

    model_config = _Frozen

    name: str
    catalog: Catalog
    memory_gb: float = Field(gt=0, description="$M$ — hard memory budget (GB)")
    t_circ_s: float = Field(gt=0, description="$T^\\circ$ — hard latency SLA (s)")
    n_d: int = Field(default=15, gt=0, description="dispatcher generated tokens")
    n_s: int = Field(default=384, gt=0, description="specialist generated tokens")
    n_y: int = Field(default=384, gt=0, description="synthesizer generated tokens")
    domains: tuple[str, ...] = Field(
        default=(), description="ordered tuple of specialist domains; e.g. MultiHop hops"
    )

    @model_validator(mode="after")
    def _non_empty(self) -> Instance:
        if not self.catalog.configs:
            raise ValueError("Instance must carry a non-empty Catalog")
        return self


class Allocation(BaseModel):
    """An assignment $(x_{r,c}, z_g)$ with its evaluated objectives.

    This is the typed artifact handed to a downstream **dynamic optimization**
    step (out of scope for this package). Always references the originating
    `Instance` by name so the consumer can rehydrate context.
    """

    model_config = _Frozen

    instance_name: str
    config_by_role: dict[Role, str] = Field(
        description="role -> Config.config_id of the chosen configuration"
    )
    loaded_groups: tuple[tuple[str, str], ...] = Field(
        description="(model_id, quant) keys with z_g = 1"
    )
    Q: float = Field(description="composite quality (objective value)")
    L_total_s: float = Field(ge=0, description="end-to-end latency under the concurrent model")
    memory_used_gb: float = Field(ge=0)
    feasible: bool
    source: str = Field(description="solver tag, e.g. 'milp', 'baseline:largest_fits'")
