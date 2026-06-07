"""HiGHS MILP — single-objective max Q under hard memory + hard SLA.

Variables:
- $x_{r,c} \\in \\{0,1\\}$ for $r\\in\\{d, s, y\\}$, $c\\in\\mathcal{K}_r$ (sparse).
- $z_g \\in \\{0,1\\}$ for $g\\in\\mathcal{G}$.
- $\\Lambda \\ge 0$ continuous (specialist latency under concurrent serving).

Objective:
    $$\\max\\ \\ F_d(c_d) + (1/|D|)\\sum_{\\delta\\in D} Q_s(c_s, \\delta) + Q_y(c_y).$$

Constraints:
- $\\sum_c x_{r,c} = 1\\quad\\forall r$.
- $x_{r,c} \\le z_{g(c)}\\quad\\forall r, c$.
- $\\sum_g w_g z_g + \\sum_{r, c} \\kappa_c x_{r,c} \\le M$.
- $\\Lambda \\ge L_{s,c}\\cdot x_{s,c}\\quad\\forall c\\in\\mathcal{K}_s$.
- $L_d(c_d) + \\Lambda + L_y(c_y) \\le T^\\circ$.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import highspy
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "highspy is required for the MILP solver. "
        "Install: pip install -r mamap_repo/lean/requirements.txt"
    ) from exc

from src.instance import InstanceArrays, build_arrays
from src.quality import Quality
from src.quality_coverage import covered_eligibility, per_config_coefficients
from src.types import Allocation, Instance, Role


@dataclass(frozen=True)
class SolveResult:
    allocation: Allocation | None
    wall_s: float
    diagnostics: dict[str, Any]


def _q_per_config(
    arrays: InstanceArrays, quality: Quality, domains: tuple[str, ...]
) -> dict[Role, np.ndarray]:
    """Pre-compute Q coefficients per (role, k) — NaN if no measurement exists.

    Delegates to :func:`src.quality_coverage.per_config_coefficients` so the
    same NaN-aware convention is used by the baselines.
    """
    return per_config_coefficients(arrays, quality, domains)


def solve_milp(
    instance: Instance,
    quality: Quality,
    time_limit_s: float | None = None,
    latency_model: str = "concurrent",
    k_active: int = 1,
) -> SolveResult:
    """Solve the lean MAMAP MILP under one of two latency models.

    Args:
        latency_model: ``"concurrent"`` (default) uses Λ as in §formulation;
            ``"sequential"`` enforces the colleague's $L_d + k\\cdot L_s + L_y$ —
            used by the §ablation-1 comparison.
        k_active: only consulted under the sequential model. Number of
            activated specialists per query.
    """
    if latency_model not in {"concurrent", "sequential"}:
        raise ValueError(f"latency_model must be concurrent|sequential; got {latency_model!r}")
    arrays = build_arrays(instance)
    t0 = time.perf_counter()

    h = highspy.Highs()
    h.silent()
    h.setOptionValue("parallel", "off")
    h.setOptionValue("random_seed", 0)
    h.setOptionValue("threads", 1)
    if time_limit_s is not None:
        h.setOptionValue("time_limit", float(time_limit_s))

    inf = highspy.kHighsInf
    n_g = arrays.weights_g.shape[0]
    q_coefs = _q_per_config(arrays, quality, instance.domains)
    covered = covered_eligibility(arrays, q_coefs)

    # ---- Variables ----
    z_var: dict[int, int] = {}
    for g in range(n_g):
        col = h.getNumCol()
        h.addCol(0.0, 0.0, 1.0, 0, [], [])
        h.changeColIntegrality(col, highspy.HighsVarType.kInteger)
        z_var[g] = col

    x_var: dict[tuple[Role, int], int] = {}
    for role in Role:
        for k in covered[role].tolist():
            col = h.getNumCol()
            cost = float(q_coefs[role][k])  # maximise Q
            h.addCol(cost, 0.0, 1.0, 0, [], [])
            h.changeColIntegrality(col, highspy.HighsVarType.kInteger)
            x_var[(role, int(k))] = col

    # Λ (continuous, nonneg).
    lambda_col = h.getNumCol()
    h.addCol(0.0, 0.0, inf, 0, [], [])

    h.changeObjectiveSense(highspy.ObjSense.kMaximize)

    # ---- Constraints ----
    # (assignment) Σ_c x_{r,c} = 1
    for role in Role:
        cols = [x_var[(role, int(k))] for k in covered[role].tolist()]
        vals = [1.0] * len(cols)
        h.addRow(1.0, 1.0, len(cols), cols, vals)

    # (load–use) x_{r,c} − z_{g(c)} ≤ 0
    for (role, k), col in x_var.items():
        g = int(arrays.group_of_k[k])
        h.addRow(-inf, 0.0, 2, [col, z_var[g]], [1.0, -1.0])

    # (memory) Σ_g w_g z_g + Σ_{r,c} κ_c x_{r,c} ≤ M
    mem_cols: list[int] = []
    mem_vals: list[float] = []
    for g in range(n_g):
        mem_cols.append(z_var[g])
        mem_vals.append(float(arrays.weights_g[g]))
    for (role, k), col in x_var.items():
        mem_cols.append(col)
        mem_vals.append(float(arrays.kv_gb[k]))
    h.addRow(-inf, float(arrays.memory_budget), len(mem_cols), mem_cols, mem_vals)

    # Latency constraints differ between the concurrent and sequential models.
    L_s = arrays.L_per_role[Role.SPECIALIST]
    L_d = arrays.L_per_role[Role.DISPATCHER]
    L_y = arrays.L_per_role[Role.SYNTHESIZER]

    if latency_model == "concurrent":
        # (concurrent cover) Λ ≥ L_{s,c} · x_{s,c}  →  −Λ + L_{s,c}·x_{s,c} ≤ 0
        for k in covered[Role.SPECIALIST].tolist():
            h.addRow(
                -inf, 0.0, 2,
                [lambda_col, x_var[(Role.SPECIALIST, int(k))]],
                [-1.0, float(L_s[int(k)])],
            )
        # (SLA hard) L_d + Λ + L_y ≤ T°
        sla_cols: list[int] = []
        sla_vals: list[float] = []
        for k in covered[Role.DISPATCHER].tolist():
            sla_cols.append(x_var[(Role.DISPATCHER, int(k))])
            sla_vals.append(float(L_d[int(k)]))
        for k in covered[Role.SYNTHESIZER].tolist():
            sla_cols.append(x_var[(Role.SYNTHESIZER, int(k))])
            sla_vals.append(float(L_y[int(k)]))
        sla_cols.append(lambda_col)
        sla_vals.append(1.0)
        h.addRow(-inf, float(arrays.t_circ_s), len(sla_cols), sla_cols, sla_vals)
    else:  # sequential: L_d + k_active·L_s + L_y ≤ T°  (Λ unused)
        h.changeColBounds(lambda_col, 0.0, 0.0)
        sla_cols = []
        sla_vals = []
        for k in covered[Role.DISPATCHER].tolist():
            sla_cols.append(x_var[(Role.DISPATCHER, int(k))])
            sla_vals.append(float(L_d[int(k)]))
        for k in covered[Role.SPECIALIST].tolist():
            sla_cols.append(x_var[(Role.SPECIALIST, int(k))])
            sla_vals.append(float(k_active) * float(L_s[int(k)]))
        for k in covered[Role.SYNTHESIZER].tolist():
            sla_cols.append(x_var[(Role.SYNTHESIZER, int(k))])
            sla_vals.append(float(L_y[int(k)]))
        h.addRow(-inf, float(arrays.t_circ_s), len(sla_cols), sla_cols, sla_vals)

    h.run()
    wall = time.perf_counter() - t0

    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return SolveResult(allocation=None, wall_s=wall, diagnostics={"status": "infeasible"})

    sol = h.getSolution()
    col_vals = sol.col_value

    chosen: dict[Role, int] = {}
    for role in Role:
        elig = covered[role].tolist()
        best_k = -1
        best_val = -1.0
        for k in elig:
            v = col_vals[x_var[(role, int(k))]]
            if v > best_val:
                best_val = v
                best_k = int(k)
        chosen[role] = best_k

    groups_used = sorted({int(arrays.group_of_k[k]) for k in chosen.values()})
    weights_part = float(arrays.weights_g[groups_used].sum())
    kv_part = float(sum(arrays.kv_gb[k] for k in chosen.values()))
    mem_used = weights_part + kv_part
    Lambda = float(col_vals[lambda_col])
    if latency_model == "concurrent":
        L_total = (
            float(L_d[chosen[Role.DISPATCHER]])
            + Lambda
            + float(L_y[chosen[Role.SYNTHESIZER]])
        )
    else:
        L_total = (
            float(L_d[chosen[Role.DISPATCHER]])
            + k_active * float(L_s[chosen[Role.SPECIALIST]])
            + float(L_y[chosen[Role.SYNTHESIZER]])
        )
    Q_value = sum(float(q_coefs[role][chosen[role]]) for role in Role)

    allocation = Allocation(
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
        source="milp",
    )
    return SolveResult(
        allocation=allocation,
        wall_s=wall,
        diagnostics={
            "status": "optimal",
            "lambda": Lambda,
            "n_groups_loaded": len(groups_used),
            "weights_gb": weights_part,
            "kv_gb": kv_part,
        },
    )
