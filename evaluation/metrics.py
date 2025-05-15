from datetime import datetime, timedelta
import pulp
from config import config
from database.dataManager import load_and_prepare_prices
from datetime import datetime
import pulp, pandas as pd
import numpy as np
from config import config
from flexoffer_logic import Flexoffer, TimeSlice, DFO


def compute_mean_runtimes(runtimes_sch, runtimes_agg):
    return {
        "runtime_scheduling": round(np.mean([runtimes_sch]) if runtimes_sch else 0.0, 3),
        "runtime_aggregation": round(np.mean([runtimes_agg]) if runtimes_agg else 0.0, 3)
    }

def compute_financial_metrics(daily_results):

    mean_sch_spot = np.mean([r["rev_sched"]["spot_rev"] for r in daily_results])
    mean_sch_res  = np.mean([r["rev_sched"]["res_rev"]  for r in daily_results])
    mean_sch_act  = np.mean([r["rev_sched"]["act_rev"]  for r in daily_results])

    mean_base_spot = np.mean([r["rev_base"]["spot_rev"] for r in daily_results])
    optimal_total  = np.mean([r["rev_opt"]["total_rev"]  for r in daily_results])

    saved_spot = mean_base_spot - mean_sch_spot
    gain_res   = mean_sch_res
    gain_act   = mean_sch_act

    pct_saved_spot = 100.0 * saved_spot / mean_base_spot if mean_base_spot else None
    pct_gain_res   = 100.0 * gain_res   / mean_base_spot if mean_base_spot else None
    pct_gain_act   = 100.0 * gain_act   / mean_base_spot if mean_base_spot else None

    total_rev      = mean_sch_res + mean_sch_act - mean_sch_spot
    pct_of_optimal = 100.0 * total_rev / optimal_total if optimal_total else None
    pct_total_saved = 100.0 * (saved_spot + gain_res + gain_act) / mean_base_spot \
                      if mean_base_spot else None

    return {
        "mean_sch_spot":   round(mean_sch_spot,  3),
        "mean_sch_res":    round(mean_sch_res,   3),
        "mean_sch_act":    round(mean_sch_act,   3),
        "mean_base_spot":  round(mean_base_spot, 3),

        "saved_spot":      round(saved_spot,     3),
        "gain_res":        round(gain_res,       3),
        "gain_act":        round(gain_act,       3),

        "pct_saved_spot":  round(pct_saved_spot, 2) if pct_saved_spot is not None else None,
        "pct_gain_res":    round(pct_gain_res,   2) if pct_gain_res is not None else None,
        "pct_gain_act":    round(pct_gain_act,   2) if pct_gain_act is not None else None,

        "total_rev":       round(total_rev,      3),
        "pct_total_saved": round(pct_total_saved,2) if pct_total_saved is not None else None,
        "pct_of_optimal":  round(pct_of_optimal, 3) if pct_of_optimal is not None else None,
    }


def greedy_baseline_schedule(offers, horizon):
    """
    Greedy baseline: always use sim_start_ts to index into prices.
    """
    sol = {k: {} for k in ("p","pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn")}
    if not offers:
        return sol

    slot_sec = config.TIME_RESOLUTION  # seconds per slot
    dt       = slot_sec / 3600.0       # hours per slot
    sim_start_ts = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    T = horizon

    for a, fo in enumerate(offers):
        max_total = fo.get_min_overall_alloc()
        prof      = fo.get_profile()
        dur       = fo.get_duration()
        start_ts  = fo.get_est()
        max_energy_per_slot = prof[0].max_power

        base_idx  = int((start_ts - sim_start_ts) / slot_sec)

        remaining = max_total
        p_dict    = {}

        # --- clamp every scheduled slot to [0, T)
        for i in range(dur):
            if remaining <= 0:
                break
            idx = base_idx + i
            # never schedule outside the simulation horizon
            if idx < 0 or idx >= T:
                break
            energy_slot = min(max_energy_per_slot, remaining)
            p_val = energy_slot / dt
            p_dict[idx] = p_val
            remaining  -= energy_slot


        sol["p"][a] = p_dict
        for key in ("pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"):
            sol[key][a] = {}

    return sol



def optimize_full_soc(
    offers: list[Flexoffer],
    spot_prices: pd.Series,
    reserve_prices: pd.DataFrame | None = None,
    activation_prices: pd.DataFrame | None = None,
    indicators: pd.DataFrame | None = None,
):
    
    # Meta‑info --------------------------------------------------------------
    res = config.TIME_RESOLUTION                  # seconds per slot
    dt  = res / 3600.0                            # hours per slot
    sim_start_ts = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    T   = len(spot_prices)                        # optimisation horizon (slots)

    A = len(offers)                               # number of charging sessions
    offsets = [int((fo.get_est() - sim_start_ts) / res) for fo in offers]

    prob = pulp.LpProblem("EV_Full_SOC", pulp.LpMaximize)

    # Decision variables -----------------------------------------------------
    p, soc = {}, {}
    pr_up, pr_dn, pb_up, pb_dn, s_up, s_dn = {}, {}, {}, {}, {}, {}

    for a, fo in enumerate(offers):
        for j, ts in enumerate(fo.get_profile()):
            t = offsets[a] + j
            if t >= T:
                continue                             # ignore if price missing

            # charging power in slice j
            p[(a, t)]   = pulp.LpVariable(f"p_{a}_{t}", lowBound=ts.min_power, upBound=ts.max_power)
            # cumulative delivered energy from arrival until end of slice j
            soc[(a, t)] = pulp.LpVariable(f"soc_{a}_{t}")

            pr_up[(a, t)] = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
            pr_dn[(a, t)] = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)
            pb_up[(a, t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
            pb_dn[(a, t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
            s_up[(a, t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
            s_dn[(a, t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    # Dynamics & bounds ------------------------------------------------------
    for a, fo in enumerate(offers):
        cap  = fo.get_max_overall_alloc()          # EV battery capacity (kWh)
        need = fo.get_min_overall_alloc()          # required energy at departure
        dur  = fo.get_duration()

        for j in range(dur):
            t = offsets[a] + j
            if t >= T:
                continue

            # SoC recursion (sum of previous power)
            if j == 0:
                prob += soc[(a, t)] == p[(a, t)] * dt, f"soc_init_{a}_{t}"
            else:
                prev_t = offsets[a] + j - 1
                if prev_t < T:
                    prob += soc[(a, t)] == soc[(a, prev_t)] + p[(a, t)] * dt, f"soc_rec_{a}_{t}"

            # battery capacity limit
            prob += soc[(a, t)] <= cap, f"soc_cap_{a}_{t}"

        last_t = offsets[a] + dur - 1
        if last_t < T:
            prob += soc[(a, last_t)] >= fo.get_min_overall_alloc(), f"soc_target_{a}"
            
    # Reserve, activation coupling ------------------------------------------
    for (a, t), _ in pr_up.items():
        prob += pr_up[(a, t)] <= p[(a, t)],                         f"ru_le_p_{a}_{t}"
        prob += pr_dn[(a, t)] <= p[(a, t)].upBound - p[(a, t)],    f"rd_le_room_{a}_{t}"

        delta_up, delta_dn = (0, 0) if indicators is None else indicators[t]
        prob += pb_up[(a, t)] + s_up[(a, t)] >= pr_up[(a, t)] * delta_up, f"su_call_{a}_{t}"
        prob += pb_dn[(a, t)] + s_dn[(a, t)] >= pr_dn[(a, t)] * delta_dn, f"sd_call_{a}_{t}"

    # Objective -------------------------------------------------------------
    expr = []
    for (a, t), var in p.items():
        expr.append(-spot_prices.iloc[t] * var * dt)
        r_up, r_dn = reserve_prices.iloc[t] if reserve_prices is not None else (0, 0)
        expr.append((r_up * pr_up[(a, t)] + r_dn * pr_dn[(a, t)]) * dt)
        b_up, b_dn = activation_prices.iloc[t] if activation_prices is not None else (0, 0)
        expr.append((b_up * pb_up[(a, t)] + b_dn * pb_dn[(a, t)]) * dt)
        expr.append(-config.PENALTY * (s_up[(a, t)] + s_dn[(a, t)]))

    prob += pulp.lpSum(expr)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract solution ------------------------------------------------------
    sol = {k: {a: {} for a in range(A)} for k in ("p", "pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn")}

    for (a, t), var in p.items():     sol["p"][a][t]      = var.value()
    for (a, t), var in pr_up.items(): sol["pr_up"][a][t]  = var.value()
    for (a, t), var in pr_dn.items(): sol["pr_dn"][a][t]  = var.value()
    for (a, t), var in pb_up.items(): sol["pb_up"][a][t]  = var.value()
    for (a, t), var in pb_dn.items(): sol["pb_dn"][a][t]  = var.value()
    for (a, t), var in s_up.items():  sol["s_up"][a][t]   = var.value()
    for (a, t), var in s_dn.items():  sol["s_dn"][a][t]   = var.value()

    return sol






def compute_profit(sol, spot, reserve, activation, indicators):
    """
    Given the LP solution dict and price series, compute:
      - spot_revenue, reserve_revenue, activation_revenue, penalty_cost
    """

    dt = config.TIME_RESOLUTION / 3600.0  # hours per slot


    spot_rev = 0.0
    res_rev  = 0.0
    act_rev  = 0.0
    pen_cost = 0.0

    for a, p_dict in sol["p"].items():
        for t, p_val in p_dict.items():

            # spot revenue
            spot_rev += p_val * spot.iloc[t] * dt

            # reserve revenue
            if config.RUN_RESERVE and reserve is not None:
                pr_up_val = sol["pr_up"][a].get(t, 0.0) or 0.0
                pr_dn_val = sol["pr_dn"][a].get(t, 0.0) or 0.0
                r_up, r_dn = reserve.iloc[t]
                res_rev += (pr_up_val * r_up + pr_dn_val * r_dn) * dt

            # activation revenue & penalty
            if config.RUN_ACTIVATION and activation is not None:
                pb_up_val = sol["pb_up"][a].get(t, 0.0) or 0.0
                pb_dn_val = sol["pb_dn"][a].get(t, 0.0) or 0.0
                b_up, b_dn = activation.iloc[t]
                act_rev += (pb_up_val * b_up + pb_dn_val * b_dn) * dt

                s_up_val = sol["s_up"][a].get(t, 0.0)
                s_dn_val = sol["s_dn"][a].get(t, 0.0)
                pen_cost += config.PENALTY * (s_up_val + s_dn_val) * dt

    total = res_rev + act_rev - spot_rev - pen_cost
    return {
        "spot_rev":  spot_rev,
        "res_rev":   res_rev,
        "act_rev":   act_rev,
        "penalty":   pen_cost,
        "total_rev": total
    }






