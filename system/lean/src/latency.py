"""Per-group concurrent latency model (MAMAP-lean §formulation).

Sequential (colleague's parts/1–3):

    L_total = L_d(c_d) + k · L_s(c_s) + L_y(c_y)

Per-group concurrent (this package, justified by the assumption that
multiple specialists sharing the same loaded `(m, q)` group execute in a
single batched pass on that loaded model):

    L_total = L_d(c_d) + Λ + L_y(c_y)
    Λ ≥ L_s,c · x_{s,c}   for every c eligible for the specialist role

Since exactly one x_{s,c} = 1, Λ collapses to the latency of the chosen
specialist config. This is linear in (x, Λ): one auxiliary continuous
variable plus |K_s| cover rows.

The functions here are **pure helpers** consumed by the MILP in L3. We keep
them separate so the constraint can be reused by the baselines and by the
sequential-vs-concurrent ablation (just swap the helper).
"""

from __future__ import annotations

import numpy as np

from src.instance import InstanceArrays
from src.types import Role


def concurrent_total_latency(
    arrays: InstanceArrays,
    k_dispatcher: int,
    k_specialist: int,
    k_synthesizer: int,
) -> float:
    """Compute L_d + Λ + L_y for an evaluated assignment.

    `k_*` are global catalog indices selected for each role. Λ is the
    specialist latency (single chosen config under concurrent serving).
    """
    L_d = float(arrays.L_per_role[Role.DISPATCHER][k_dispatcher])
    L_s = float(arrays.L_per_role[Role.SPECIALIST][k_specialist])
    L_y = float(arrays.L_per_role[Role.SYNTHESIZER][k_synthesizer])
    return L_d + L_s + L_y


def sequential_total_latency(
    arrays: InstanceArrays,
    k_dispatcher: int,
    k_specialist: int,
    k_synthesizer: int,
    k_active: int = 1,
) -> float:
    """Colleague's baseline: L_d + k_active · L_s + L_y.

    Provided so the MILP can build the sequential-vs-concurrent ablation
    without recomputing per-row numbers.
    """
    L_d = float(arrays.L_per_role[Role.DISPATCHER][k_dispatcher])
    L_s = float(arrays.L_per_role[Role.SPECIALIST][k_specialist])
    L_y = float(arrays.L_per_role[Role.SYNTHESIZER][k_synthesizer])
    return L_d + k_active * L_s + L_y


def specialist_latency_terms(arrays: InstanceArrays) -> np.ndarray:
    """Return $L_{s,c}$ for every catalog index — used by the MILP to build the
    `Λ ≥ L_{s,c} · x_{s,c}` cover rows.
    """
    return arrays.L_per_role[Role.SPECIALIST]


def lambda_lower_bound(
    arrays: InstanceArrays, role_assignment: dict[Role, int]
) -> float:
    """Return the value of Λ given an integer assignment — i.e. the chosen
    specialist's latency. Convenience for unit tests + baseline evaluation.
    """
    k_s = role_assignment[Role.SPECIALIST]
    return float(arrays.L_per_role[Role.SPECIALIST][k_s])
