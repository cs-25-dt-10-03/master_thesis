from datetime import datetime, timedelta
import pandas as pd
import pulp
from config import config
from optimization.flexOfferOptimizer import DFO
# Unified price loader
from database.dataManager import load_and_prepare_prices


def optimize_dfos_theoretical(
    dfos: list[DFO],
    start_date: str,
    simulation_days: int
) -> dict:
    """
    True day-ahead optimum for raw DFOs using full mFRR market model:
    - Use historical spot prices, reserve prices and real activation calls
    - Enforce capacity + activation coupling exactly
    - Respect slice polygon bounds and global energy caps
    - No clustering or relaxations
    Returns solution dict compatible with compute_profit
    """
    # Determine horizon
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = simulation_days * slots_per_day

    # Load prices and activation/call indicators
    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start_date,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )
    # activation DataFrame in this context contains actual call volumes in 'up','dn'

    slot_sec = config.TIME_RESOLUTION
    dt       = slot_sec / 3600.0
    T        = len(spot)

    # Align DFOs into this horizon
    sim0_ts  = datetime.timestamp(pd.to_datetime(start_date))
    offsets  = [int((d.get_est() - sim0_ts)/slot_sec) for d in dfos]
    A        = len(dfos)

    # Build LP
    prob = pulp.LpProblem("DFO_TrueOptimum", pulp.LpMaximize)

    # --- Decision variables ---
    p     = {}
    pr_up = {}
    pr_dn = {}
    s_up  = {}
    s_dn  = {}
    for a, d in enumerate(dfos):
        for j, poly in enumerate(d.polygons):
            t = offsets[a] + j
            if t < 0 or t >= T:
                continue
            # charging power
            p[(a,t)]     = pulp.LpVariable(f"p_{a}_{t}",     lowBound=0)
            # reserve capacity
            pr_up[(a,t)] = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
            pr_dn[(a,t)] = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)
            # actual activation
            s_up[(a,t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
            s_dn[(a,t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    # --- Objective: spot cost (negative), reserve capacity revenue, activation energy revenue ---
    terms = []
    for (a,t), var in p.items():
        terms.append(- spot.iloc[t] * var * dt)
    for (a,t), var in pr_up.items():
        terms.append(reserve['up'].iloc[t] * var * dt)
    for (a,t), var in pr_dn.items():
        terms.append(reserve['dn'].iloc[t] * var * dt)
    for (a,t), var in s_up.items():
        terms.append(activation['up'].iloc[t] * var * dt)
    for (a,t), var in s_dn.items():
        terms.append(activation['dn'].iloc[t] * var * dt)
    prob += pulp.lpSum(terms)

    # --- Constraints ---
    # 1) Polygon bounds on p
    for (a,t), var in p.items():
        poly = dfos[a].polygons[t-offsets[a]]
        ys = [pt.y for pt in poly.points]
        prob += var >= min(ys), f"slice_min_{a}_{t}"
        prob += var <= max(ys), f"slice_max_{a}_{t}"

    # 2) Reserve coupling
    for (a,t), rvar in pr_up.items():
        prob += rvar <= p[(a,t)], f"ru_le_p_{a}_{t}"
    for (a,t), rvar in pr_dn.items():
        poly = dfos[a].polygons[t-offsets[a]]
        p_max = max(pt.y for pt in poly.points)
        prob += rvar <= p_max - p[(a,t)], f"rd_le_cap_{a}_{t}"

    # 3) Activation coupling and actual calls
    for (a,t), svar in s_up.items():
        prob += svar <= pr_up[(a,t)], f"su_le_ru_{a}_{t}"
        prob += svar <= indicators['up'].iloc[t], f"su_le_call_{a}_{t}"
    for (a,t), svar in s_dn.items():
        prob += svar <= pr_dn[(a,t)], f"sd_le_rd_{a}_{t}"
        prob += svar <= indicators['dn'].iloc[t], f"sd_le_call_{a}_{t}"

    # 4) Global energy caps
    for a, d in enumerate(dfos):
        energy_terms = []
        for j in range(len(d.polygons)):
            t = offsets[a] + j
            if 0 <= t < T:
                energy_terms.append(p[(a,t)] * dt)
        prob += pulp.lpSum(energy_terms) >= d.min_total_energy, f"min_tot_{a}"
        prob += pulp.lpSum(energy_terms) <= d.max_total_energy, f"max_tot_{a}"

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # --- Extract solution dict ---
    sol = {k:{i:{} for i in range(A)} for k in ("p","pr_up","pr_dn","s_up","s_dn")}
    for (a,t), var in p.items():      sol["p"][a][t]     = var.value()
    for (a,t), var in pr_up.items():  sol["pr_up"][a][t] = var.value()
    for (a,t), var in pr_dn.items():  sol["pr_dn"][a][t] = var.value()
    for (a,t), var in s_up.items():   sol["s_up"][a][t]  = var.value()
    for (a,t), var in s_dn.items():   sol["s_dn"][a][t]  = var.value()

    return sol


def schedule_dfos_theoretical(
    dfos: list[DFO]
) -> dict:
    """
    Wrapper to run theoretical optimization on raw DFOs.
    """
    return optimize_dfos_theoretical(
        dfos,
        start_date=config.SIMULATION_START_DATE,
        simulation_days=config.SIMULATION_DAYS
    )
