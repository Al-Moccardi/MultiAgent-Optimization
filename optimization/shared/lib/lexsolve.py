"""
lexsolve.py -- canonical solution selection for the MAMAP MILPs.

A MILP guarantees a unique optimal OBJECTIVE VALUE, but the optimal SOLUTION
(the argmax) can be a set: e.g. two quantizations of the same model with equal
objective contribution. Which member CBC returns depends on the solver build,
so allocation-level fields (chosen quant/context, hence reported latency and
memory) were not machine-independent in v1 of this package.

Fix: after the primary solve, refine lexicographically over the optimal face:
    stage 1  maximise the original objective            -> Z*
    stage 2  s.t. objective >= Z* - TOL, minimise system latency  -> L*
    stage 3  s.t. latency  <= L* + TOL, minimise loaded memory
This selects ONE canonical optimum, independent of the CBC build. TOL = 1e-6
(absolute; quality objectives are O(1) and genuinely distinct optima differ by
>= 1e-4 on this data, so the floor cannot admit a strictly worse model choice;
reported objective values are in any case recomputed in Python from the chosen
configuration, never read back from the solver).

If a refinement stage fails (time limit), the previous stage's incumbent is
restored, so the function can only improve determinism, never correctness.
"""
import pulp

TOL = 1e-6


def lex_refine(pr, obj_expr, lat_expr, mem_expr, tl=None, extra=None):
    """Refine the already-solved problem `pr` in place. `obj_expr` must be the
    expression whose value the stage-1 solve maximised; `lat_expr`/`mem_expr`
    are the system-latency and loaded-memory expressions of the same model.

    `extra`: optional constraints applied before refinement. The gated
    (McCormick) scripts pass model-fixing constraints here (the stage-1 model
    triple is kept; only the configuration within each model is canonicalized),
    because refining over the full optimal face of the bilinear model is slow
    for CBC. The objective pins the model triple up to exact float ties of
    measured means, which do not occur in this data; a hypothetical exact
    model-level tie would still follow solver order and is the one residual
    non-determinism, stated here explicitly.
    """
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=tl) if tl else pulp.PULP_CBC_CMD(msg=0)
    z_star = pulp.value(obj_expr)
    snap = {v.name: v.value() for v in pr.variables()}
    for i, cn in enumerate(extra or []):
        pr += cn, f"lex_fix_{i}"

    pr += obj_expr >= z_star - TOL, "lex_obj_floor"
    pr.sense = pulp.LpMinimize
    pr.setObjective(lat_expr)
    pr.solve(solver)
    if pulp.LpStatus[pr.status] != "Optimal":
        for v in pr.variables():
            v.varValue = snap[v.name]
        return
    l_star = pulp.value(lat_expr)
    snap = {v.name: v.value() for v in pr.variables()}

    pr += lat_expr <= l_star + TOL, "lex_lat_cap"
    pr.setObjective(mem_expr)
    pr.solve(solver)
    if pulp.LpStatus[pr.status] != "Optimal":
        for v in pr.variables():
            v.varValue = snap[v.name]
