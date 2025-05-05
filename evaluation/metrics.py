
import pulp
from config import config
from flexoffer_logic import DFO, Flexoffer

def theo_opt(dfos, spot, reserve=None, activation=None):
    """
    Perfect-foresight LP relaxation bound over unaggregated DFOs.
    Returns (solution_dict, bound_value).
    """

    dt = config.TIME_RESOLUTION / 3600.0  # hours per slot
    model = pulp.LpProblem("FullMarket_LPBound", pulp.LpMaximize)

    # Decision variables
    p    = {}  # spot
    pru  = {}  # reserve up bids
    prd  = {}  # reserve down bids
    pbu  = {}  # activation up
    pbd  = {}  # activation down
    E    = {}  # cumulative energy

    # Build the LP
    for v, dfo in enumerate(dfos):
        # initialize cumulative energy at 0
        E[v,0] = pulp.LpVariable(f"E_{v}_0", lowBound=0, upBound=0)
        model += E[v,0] == 0

        for i, poly in enumerate(dfo.polygons):
            # map slice i to global time index t
            t = int((dfo.start_ts + i*config.TIME_RESOLUTION - dfo.global_start) 
                    / config.TIME_RESOLUTION)

            # create continuous vars
            p[v,i]   = pulp.LpVariable(f"p_{v}_{i}",   lowBound=0, upBound=dfo.charging_power)
            pru[v,i] = pulp.LpVariable(f"pru_{v}_{i}", lowBound=0, upBound=dfo.charging_power)
            prd[v,i] = pulp.LpVariable(f"prd_{v}_{i}", lowBound=0, upBound=dfo.charging_power)
            pbu[v,i] = pulp.LpVariable(f"pbu_{v}_{i}", lowBound=0, upBound=dfo.charging_power)
            pbd[v,i] = pulp.LpVariable(f"pbd_{v}_{i}", lowBound=0, upBound=dfo.charging_power)

            # convex‐combination of polygon vertices
            # this enforces (E_prev, p) ∈ polygon
            ws = []
            for j, (e_j, p_j) in enumerate(poly.points):
                w = pulp.LpVariable(f"w_{v}_{i}_{j}", lowBound=0, upBound=1)
                ws.append(w)
                # accumulate objective contribution from spot
                # we'll add full objective below
            model += pulp.lpSum(ws) == 1
            model += E[v,i] == pulp.lpSum(ws[j]*poly.points[j][0] for j in range(len(ws)))
            model += p[v,i] == pulp.lpSum(ws[j]*poly.points[j][1] for j in range(len(ws)))

            # energy evolution
            E[v,i+1] = pulp.LpVariable(f"E_{v}_{i+1}", lowBound=0)
            model += E[v,i+1] == E[v,i] + p[v,i]*dt

            # power‐coupling constraints
            model += p[v,i] + pru[v,i] <= dfo.charging_power
            model += p[v,i] - prd[v,i] >= 0
            model += pbu[v,i] <= pru[v,i]
            model += pbd[v,i] <= prd[v,i]

            # accumulate into objective
            # spot
            model += spot[t] * p[v,i] * dt
            # reserve (if enabled)
            if reserve is not None:
                model += reserve["up"][t]   * pru[v,i] * dt
                model += reserve["dn"][t]   * prd[v,i] * dt
            # activation (if enabled)
            if activation is not None:
                model += activation["up"][t] * pbu[v,i] * dt
                model += activation["dn"][t] * pbd[v,i] * dt

    # Solve **as a pure LP** (no integers)
    model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    # pull out the bound
    bound_value = pulp.value(model.objective)

    # (optional) you can extract a “solution_dict” in the same shape as schedule_offers
    solution = {"p": {}, "pr_up": {}, "pr_dn": {}, "pb_up": {}, "pb_dn": {}, "s_up": {}, "s_dn": {}}
    for v, dfo in enumerate(dfos):
        solution["p"][v] = {i: p[v,i].varValue for i in range(len(dfo.polygons))}
        solution["pr_up"][v] = {i: pru[v,i].varValue for i in range(len(dfo.polygons))}
        solution["pr_dn"][v] = {i: prd[v,i].varValue for i in range(len(dfo.polygons))}
        solution["pb_up"][v] = {i: pbu[v,i].varValue for i in range(len(dfo.polygons))}
        solution["pb_dn"][v] = {i: pbd[v,i].varValue for i in range(len(dfo.polygons))}
    return solution, bound_value
