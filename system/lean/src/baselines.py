"""Naive role-aware baselines (lean §results).

Same four shapes as main MAMAP, adapted from per-agent to per-role:

- **largest-fits**: pick the **largest** (max params) single Qwen `(m, q)`
  config that fits in $M$ and respects $T^\\circ$ across all three roles
  using one *shared* group. Replicate it everywhere.
- **per-role-best**: per role, pick the config maximising the role's Q
  coefficient, ignoring shared memory. Reports `feasible=False` if the
  joint allocation overflows $M$.
- **uniform**: same config across all three roles; pick the one with the
  highest total Q that fits.
- **random-feasible**: sample feasible $(x, z)$ uniformly at random; return
  the one with the highest Q over `n_samples` draws.

All return `Allocation` objects with `source="baseline:..."`.
"""

from __future__ import annotations

import numpy as np

from src.instance import InstanceArrays, build_arrays
from src.quality import Quality
from src.quality_coverage import covered_eligibility, per_config_coefficients
from src.types import Allocation, Instance, Role

_ROLES_ORDER = (Role.DISPATCHER, Role.SPECIALIST, Role.SYNTHESIZER)


def _q_per_config(
    arrays: InstanceArrays, quality: Quality, domains: tuple[str, ...]
) -> dict[Role, np.ndarray]:
    """NaN-aware delegate to keep baselines and MILP in sync."""
    return per_config_coefficients(arrays, quality, domains)


def _make_allocation(
    instance: Instance,
    arrays: InstanceArrays,
    chosen: dict[Role, int],
    q_coefs: dict[Role, np.ndarray],
    source: str,
) -> Allocation:
    groups_used = sorted({int(arrays.group_of_k[k]) for k in chosen.values()})
    weights_part = float(arrays.weights_g[groups_used].sum())
    kv_part = float(sum(arrays.kv_gb[k] for k in chosen.values()))
    mem_used = weights_part + kv_part
    Lambda = float(arrays.L_per_role[Role.SPECIALIST][chosen[Role.SPECIALIST]])
    L_total = (
        float(arrays.L_per_role[Role.DISPATCHER][chosen[Role.DISPATCHER]])
        + Lambda
        + float(arrays.L_per_role[Role.SYNTHESIZER][chosen[Role.SYNTHESIZER]])
    )
    Q_value = sum(float(q_coefs[role][chosen[role]]) for role in Role)
    return Allocation(
        instance_name=instance.name,
        config_by_role={role: arrays.config_ids[chosen[role]] for role in Role},
        loaded_groups=tuple(arrays.group_keys[g] for g in groups_used),
        Q=Q_value,
        L_total_s=L_total,
        memory_used_gb=mem_used,
        feasible=(
            mem_used <= arrays.memory_budget + 1e-6
            and L_total <= arrays.t_circ_s + 1e-6
        ),
        source=source,
    )


def _intersect_eligibility(
    arrays: InstanceArrays, covered: dict[Role, np.ndarray]
) -> list[int]:
    """Configs eligible (SLA + quality) for *every* role — needed by
    largest-fits and uniform which assign the same config to all three roles."""
    common = set(covered[Role.DISPATCHER].tolist())
    for role in (Role.SPECIALIST, Role.SYNTHESIZER):
        common &= set(covered[role].tolist())
    return sorted(common)


def solve_largest_fits(
    instance: Instance, quality: Quality
) -> Allocation | None:
    """Pick the largest single `(m, q)` config (max params) that fits across roles."""
    arr = build_arrays(instance)
    q = _q_per_config(arr, quality, instance.domains)
    covered = covered_eligibility(arr, q)
    candidates: list[int] = []
    for k in _intersect_eligibility(arr, covered):
        # Memory: load weights once (k's group) + κ_k for each of 3 roles
        g = int(arr.group_of_k[k])
        mem = float(arr.weights_g[g]) + 3 * float(arr.kv_gb[k])
        L_total = (
            float(arr.L_per_role[Role.DISPATCHER][k])
            + float(arr.L_per_role[Role.SPECIALIST][k])
            + float(arr.L_per_role[Role.SYNTHESIZER][k])
        )
        if mem <= arr.memory_budget + 1e-6 and L_total <= arr.t_circ_s + 1e-6:
            candidates.append(k)
    if not candidates:
        return None
    # Largest = max params
    config_params = np.array(
        [instance.catalog.configs[k].group.params for k in candidates], dtype=np.int64
    )
    best_k = candidates[int(np.argmax(config_params))]
    chosen = dict.fromkeys(_ROLES_ORDER, best_k)
    return _make_allocation(instance, arr, chosen, q, source="baseline:largest_fits")


def solve_per_role_best(instance: Instance, quality: Quality) -> Allocation:
    arr = build_arrays(instance)
    q = _q_per_config(arr, quality, instance.domains)
    covered = covered_eligibility(arr, q)
    chosen: dict[Role, int] = {}
    for role in _ROLES_ORDER:
        elig = covered[role]
        best_local = int(np.argmax(q[role][elig]))
        chosen[role] = int(elig[best_local])
    return _make_allocation(instance, arr, chosen, q, source="baseline:per_role_best")


def solve_uniform(instance: Instance, quality: Quality) -> Allocation | None:
    """Same config across all roles; pick the one with maximum total Q that fits."""
    arr = build_arrays(instance)
    q = _q_per_config(arr, quality, instance.domains)
    covered = covered_eligibility(arr, q)
    common = _intersect_eligibility(arr, covered)
    if not common:
        return None
    best_k: int | None = None
    best_Q: float = -np.inf
    for k in common:
        g = int(arr.group_of_k[k])
        mem = float(arr.weights_g[g]) + 3 * float(arr.kv_gb[k])
        L_total = (
            float(arr.L_per_role[Role.DISPATCHER][k])
            + float(arr.L_per_role[Role.SPECIALIST][k])
            + float(arr.L_per_role[Role.SYNTHESIZER][k])
        )
        if mem > arr.memory_budget + 1e-6 or L_total > arr.t_circ_s + 1e-6:
            continue
        Q_total = float(q[Role.DISPATCHER][k] + q[Role.SPECIALIST][k] + q[Role.SYNTHESIZER][k])
        if Q_total > best_Q:
            best_Q = Q_total
            best_k = k
    if best_k is None:
        return None
    chosen = dict.fromkeys(_ROLES_ORDER, best_k)
    return _make_allocation(instance, arr, chosen, q, source="baseline:uniform")


def solve_random_feasible(
    instance: Instance,
    quality: Quality,
    n_samples: int = 500,
    seed: int = 0,
) -> Allocation | None:
    """Sample $n$ feasible allocations; return the best."""
    arr = build_arrays(instance)
    q = _q_per_config(arr, quality, instance.domains)
    covered = covered_eligibility(arr, q)
    rng = np.random.default_rng(seed)
    best: Allocation | None = None
    for _ in range(n_samples):
        chosen = {
            role: int(rng.choice(covered[role])) for role in _ROLES_ORDER
        }
        alloc = _make_allocation(instance, arr, chosen, q, source="baseline:random_feasible")
        if alloc.feasible and (best is None or alloc.Q > best.Q):
            best = alloc
    return best
