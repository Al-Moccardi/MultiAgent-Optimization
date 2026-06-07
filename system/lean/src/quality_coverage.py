"""Per-role quality coefficients with NaN where unmeasured.

Pulled into its own module because both `src.milp` and `src.baselines` need
the same filtering logic, and we want NaN (not zero) to flag "no measured
quality available for this (role, config[, domain])" — so the solver can
*skip* those configs instead of treating them as bad-but-pickable.
"""

from __future__ import annotations

import numpy as np

from src.instance import InstanceArrays
from src.quality import Quality
from src.types import Role


def per_config_coefficients(
    arrays: InstanceArrays,
    quality: Quality,
    domains: tuple[str, ...],
) -> dict[Role, np.ndarray]:
    """Compute ``q[role][k]`` with NaN where no measurement exists.

    - Role ``d``  → ``F_d[cid]`` or NaN.
    - Role ``y``  → ``Q_y[cid]`` or NaN.
    - Role ``s``  → mean of ``Q_s[(cid, δ)]`` over the *measured* domains
      ⊆ ``domains``. NaN if no domain is measured.
    """
    n_k = arrays.weight_gb_of_k.shape[0]
    out: dict[Role, np.ndarray] = {}
    for role in Role:
        coefs = np.full(n_k, np.nan, dtype=np.float64)
        if role is Role.DISPATCHER:
            for k, cid in enumerate(arrays.config_ids):
                if cid in quality.F_d:
                    coefs[k] = float(quality.F_d[cid])
        elif role is Role.SYNTHESIZER:
            for k, cid in enumerate(arrays.config_ids):
                if cid in quality.Q_y:
                    coefs[k] = float(quality.Q_y[cid])
        else:  # SPECIALIST
            ds = domains or ("default",)
            for k, cid in enumerate(arrays.config_ids):
                measured = [
                    float(quality.Q_s[(cid, d)])
                    for d in ds
                    if (cid, d) in quality.Q_s
                ]
                if measured:
                    coefs[k] = float(sum(measured) / len(measured))
        out[role] = coefs
    return out


def covered_eligibility(
    arrays: InstanceArrays,
    q_coefs: dict[Role, np.ndarray],
) -> dict[Role, np.ndarray]:
    """Intersect the SLA-pre-filtered ``arrays.eligibility`` with quality coverage.

    Configs whose ``q_coefs[role][k]`` is NaN are dropped from $K_r$. Raises
    ``ValueError`` if any role ends up with an empty $K_r$ — the user must
    relax the SLA, broaden the catalog, or add quality measurements.
    """
    out: dict[Role, np.ndarray] = {}
    for role in Role:
        elig = arrays.eligibility[role]
        if elig.size == 0:
            raise ValueError(f"Role '{role.value}' has empty SLA eligibility before quality filter")
        coefs = q_coefs[role]
        keep = np.array(
            [int(k) for k in elig.tolist() if not np.isnan(coefs[int(k)])],
            dtype=np.int64,
        )
        if keep.size == 0:
            raise ValueError(
                f"Role '{role.value}' has empty K_r after quality coverage filter. "
                f"All SLA-eligible configs lack measured quality — relax SLA, "
                f"broaden the catalog, or fill the quality table."
            )
        out[role] = keep
    return out
