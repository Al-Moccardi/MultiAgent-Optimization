"""Instance loader + per-role eligibility (SLA pre-filter).

Mirrors `c:\\Users\\mfoni\\Desktop\\MAMAP\\src\\mamap\\instance.py` (parent
MAMAP) but with three roles instead of N agents. The SLA pre-filter is the
key design choice carried over: configs with $L_c$ alone exceeding $T^\\circ$
never enter $\\mathcal{K}_r$, so the MILP/baselines can't produce
SLA-violating rows by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml
from catalog.build_catalog import read_json

from src.types import Catalog, Instance, Role


@dataclass(frozen=True)
class InstanceArrays:
    """Numpy mirror of an Instance — the hot path for solvers."""

    # Per-config (length |K|)
    weight_gb_of_k: npt.NDArray[np.float64]  # w_{g(k)} (repeated per group)
    kv_gb: npt.NDArray[np.float64]           # κ_k
    L_per_role: dict[Role, npt.NDArray[np.float64]]  # role -> L_k under that role's n_gen
    E_per_role: dict[Role, npt.NDArray[np.float64]]  # role -> per-call energy
    group_of_k: npt.NDArray[np.int64]        # global group index per config
    config_ids: tuple[str, ...]              # stable ids matching the perf/quality tables

    # Per-group (length |G|)
    weights_g: npt.NDArray[np.float64]       # w_g (one number per group)
    group_keys: tuple[tuple[str, str], ...]  # (model_id, quant) per group index

    # Per-role eligibility (after SLA pre-filter)
    eligibility: dict[Role, npt.NDArray[np.int64]]  # role -> global k indices in K_r

    # Globals
    memory_budget: float
    t_circ_s: float
    n_gen_per_role: dict[Role, int]


def _build_group_index(catalog: Catalog) -> tuple[list, dict[tuple[str, str], int]]:
    groups = list(catalog.groups)
    return groups, {g.key: i for i, g in enumerate(groups)}


def build_arrays(instance: Instance) -> InstanceArrays:
    """Compute the InstanceArrays from a frozen Instance.

    Per-role eligibility filters by:
    - `ctx ≥ ?` — not applied; the lean version uses the full catalog ctx range
      (the user clips per-agent context at the MAMAP level; not relevant for
      the role-aggregated lean version).
    - `L_c ≤ T°` — applied. The hard SLA filter.
    """
    catalog = instance.catalog
    n_k = len(catalog)
    groups, group_index = _build_group_index(catalog)

    weights_g = np.array([g.weight_gb for g in groups], dtype=np.float64)
    group_keys = tuple(g.key for g in groups)
    weight_gb_of_k = np.empty(n_k, dtype=np.float64)
    kv_gb = np.empty(n_k, dtype=np.float64)
    group_of_k = np.empty(n_k, dtype=np.int64)
    config_ids: list[str] = []

    for k, cfg in enumerate(catalog.configs):
        weight_gb_of_k[k] = cfg.group.weight_gb
        kv_gb[k] = cfg.kv_gb
        group_of_k[k] = group_index[cfg.group.key]
        config_ids.append(cfg.config_id)

    n_gen = {
        Role.DISPATCHER: instance.n_d,
        Role.SPECIALIST: instance.n_s,
        Role.SYNTHESIZER: instance.n_y,
    }
    L_per_role: dict[Role, npt.NDArray[np.float64]] = {}
    E_per_role: dict[Role, npt.NDArray[np.float64]] = {}
    for role, n in n_gen.items():
        L_per_role[role] = np.array(
            [cfg.latency(n) for cfg in catalog.configs], dtype=np.float64
        )
        E_per_role[role] = np.array(
            [cfg.energy(n) for cfg in catalog.configs], dtype=np.float64
        )

    # SLA pre-filter per role. Use the role's own L array since n_gen differs.
    t_circ = instance.t_circ_s
    eligibility: dict[Role, npt.NDArray[np.int64]] = {}
    for role in Role:
        mask = L_per_role[role] <= t_circ + 1e-9
        idx = np.flatnonzero(mask).astype(np.int64)
        if idx.size == 0:
            raise ValueError(
                f"Role '{role.value}' has empty K_r at T_circ={t_circ}s "
                f"(n_gen={n_gen[role]}); relax T_circ or shrink n_gen."
            )
        eligibility[role] = idx

    return InstanceArrays(
        weight_gb_of_k=weight_gb_of_k,
        kv_gb=kv_gb,
        L_per_role=L_per_role,
        E_per_role=E_per_role,
        group_of_k=group_of_k,
        config_ids=tuple(config_ids),
        weights_g=weights_g,
        group_keys=group_keys,
        eligibility=eligibility,
        memory_budget=instance.memory_gb,
        t_circ_s=t_circ,
        n_gen_per_role=n_gen,
    )


def load_instance(yaml_path: Path) -> Instance:
    """Load a lean experiment YAML and the `catalog.json` it points to."""
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError(f"Experiment YAML must be a mapping: {yaml_path}")

    catalog_rel = spec.get("catalog")
    if not catalog_rel:
        raise ValueError(f"Missing `catalog:` field in {yaml_path}")
    lean_root = yaml_path.resolve().parent.parent
    catalog_path = (lean_root / catalog_rel).resolve()
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Catalog '{catalog_path}' not found. "
            f"Run: `python -m catalog.build_catalog`."
        )
    catalog = read_json(catalog_path)

    return Instance(
        name=spec.get("name", yaml_path.stem),
        catalog=catalog,
        memory_gb=float(spec["memory_gb"]),
        t_circ_s=float(spec["t_circ_s"]),
        n_d=int(spec.get("n_d", 15)),
        n_s=int(spec.get("n_s", 384)),
        n_y=int(spec.get("n_y", 384)),
        domains=tuple(spec.get("domains", ())),
    )
