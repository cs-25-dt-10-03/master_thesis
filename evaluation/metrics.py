from datetime import datetime, timedelta
import pulp
from config import config
from database.dataManager import load_and_prepare_prices
from datetime import datetime
import pulp, pandas as pd
import numpy as np
from config import config
from flexoffer_logic import Flexoffer, TimeSlice, DFO

from flexoffer_logic import TimeSlice, Flexoffer
from datetime import datetime
from config import config
from helpers import dt_to_unix
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value
import pandas as pd
from classes.electricVehicle import ElectricVehicle



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

    total_savings = saved_spot + gain_res + gain_act

    pct_saved_spot= 100 * saved_spot / total_savings
    pct_gain_res  = 100 * gain_res / total_savings
    pct_gain_act  = 100 * gain_act / total_savings

    total_rev = mean_sch_res + mean_sch_act - mean_sch_spot

    if optimal_total < 0 and abs(total_rev) > 1e-6:
        pct_of_optimal = 100.0 * optimal_total / total_rev
    elif abs(total_rev) <= 1e-6:
        print("[WARNING] Scheduler returned 0 total revenue — cannot compute pct_of_optimal.")
        pct_of_optimal = None
    else:
        print("[WARNING] Optimal revenue is positive — check theoretical_optimal_schedule validity.")
        pct_of_optimal = None

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



def theoretical_optimal_schedule(offers, spot, reserve, activation, indicators):

    dt = config.TIME_RESOLUTION / 3600.0
    res = config.TIME_RESOLUTION
    sim_start_ts = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    T = len(spot)
    A = len(offers)
    offsets = [int((fo.get_est() - sim_start_ts) / res) for fo in offers]

    # Decision variables
    p, pr_up, pr_dn = {}, {}, {}
    pb_up, pb_dn = {}, {}
    s_up, s_dn = {}, {}

    prob = LpProblem("TheoreticalEVRevenue", LpMaximize)

    z = {}  # z[a,s]: binary start time selector
    allowed_starts = {}  # Map of allowed start times for each offer
    for a, fo in enumerate(offers):
        allowed = fo.get_allowed_start_times()
        allowed_starts[a] = allowed
        for s_idx, s in enumerate(allowed):
            z[(a, s_idx)] = pulp.LpVariable(f"z_{a}_{s_idx}", cat='Binary')


    p = {}
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        allowed = allowed_starts[a]
        for s_idx, s in enumerate(allowed):
            for j, ts in enumerate(prof):
                t = int((s - sim_start_ts) / res) + j
                if t < T:
                    # Create power variable
                    var = pulp.LpVariable(f"p_{a}_{s_idx}_{t}", lowBound=0, upBound=ts.max_power)
                    p[(a, s_idx, t)] = var

                    # Add conditional bounds linked to binary start selection
                    prob += var <= ts.max_power * z[(a, s_idx)], f"max_bound_{a}_{s_idx}_{j}"
                    prob += var >= ts.min_power * z[(a, s_idx)], f"min_bound_{a}_{s_idx}_{j}"


    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        for j, ts in enumerate(prof):
            t = offsets[a] + j
            if t < T:
                pr_up[(a, t)] = LpVariable(f"pr_up_{a}_{t}", lowBound=0)
                pr_dn[(a, t)] = LpVariable(f"pr_dn_{a}_{t}", lowBound=0)
                pb_up[(a, t)] = LpVariable(f"pb_up_{a}_{t}", lowBound=0)
                pb_dn[(a, t)] = LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
                s_up[(a, t)] = LpVariable(f"s_up_{a}_{t}", lowBound=0)
                s_dn[(a, t)] = LpVariable(f"s_dn_{a}_{t}", lowBound=0)

    # Objective
    obj = []
    for (a, s_idx, t), var in p.items():
        if (a, t) in pr_up and (a, t) in pr_dn:
            spot_price = spot.iloc[t]
            r_up, r_dn, _ , _ = reserve.iloc[t]
            b_up, b_dn, penalty, _, _ = activation.iloc[t]
            d_up, d_dn = indicators[t]
            b_up = b_up if d_up == 1 else 0
            b_dn = b_dn if d_dn == 1 else 0

            obj.append(-spot_price * var * dt)
            obj.append(r_up * pr_up[(a, t)] * dt + r_dn * pr_dn[(a, t)] * dt)

            obj.append((b_up - spot) * pb_up[(a,t)] * dt)
            obj.append((b_dn - spot) * pb_dn[(a,t)] * dt)
            obj.append(-penalty * (s_up[(a,t)] + s_dn[(a,t)]) * dt)
    prob += lpSum(obj)

    # Constraints
    # --- only one start time ---
    for a in range(len(offers)):
        prob += pulp.lpSum(z[(a, s_idx)] for s_idx in range(len(allowed_starts[a]))) == 1



    # --- min total energy ---
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        min_energy = []
        max_energy = []
        for s_idx, s in enumerate(allowed_starts[a]):
            for j in range(len(prof)):
                t = int((s - sim_start_ts) / res) + j
                if t < T and (a, s_idx, t) in p:
                    min_energy.append(p[(a, s_idx, t)] * dt)
                    max_energy.append(p[(a, s_idx, t)] * dt)
        prob += pulp.lpSum(min_energy) >= fo.get_min_overall_alloc(), f"total_min_energy_{a}"
        prob += pulp.lpSum(max_energy) <= fo.get_max_overall_alloc(), f"total_max_energy_{a}"

        for j, ts in enumerate(prof):
            t = offsets[a] + j
            if t < T:
                # Reserve coupling
                p_total = pulp.lpSum(p[(a, s_idx, t)] for s_idx in range(len(allowed_starts[a])) if (a, s_idx, t) in p)
                prob += pr_dn[(a,t)] <= ts.max_power - p_total
                prob += pr_up[(a,t)] <= ts.max_power

                # Activation constraints (only if activated in data)
                d_up, d_dn = indicators[t]
                if d_up == 0:
                    prob += pb_up[(a,t)] == 0
                if d_dn == 0:
                    prob += pb_dn[(a,t)] == 0

                #activation coupled to reserve
                prob += pb_up[(a,t)] <= pr_up[(a,t)]
                prob += pb_dn[(a,t)] <= pr_dn[(a,t)]

                #If we have more activation than reserve, we put the overbid part in a slack variable s which is penalized
                prob += pb_up[(a, t)] + s_up[(a, t)] >= pr_up[(a, t)] * d_up
                prob += pb_dn[(a, t)] + s_dn[(a, t)] >= pr_dn[(a, t)] * d_dn
                
                #ramp constraint
                if j > 0:
                    t_prev = offsets[a] + j - 1
                    if (a, t) in pr_up and (a, t_prev) in pr_up:
                        dt_minutes = config.TIME_RESOLUTION / 60
                        ramp_limit_up = ts.max_power * (10 / dt_minutes)
                        ramp_limit_dn = ts.max_power * (10 / dt_minutes)
                        prob += pr_up[(a, t)] - pr_up[(a, t_prev)] <= ramp_limit_up
                        prob += pr_dn[(a, t)] - pr_dn[(a, t_prev)] <= ramp_limit_dn

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Pack solution
    sol = {k: {a: {} for a in range(A)} for k in ("p", "pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn")}
    for a in range(A):
        sol["p"][a] = {}
        for s_idx, s in enumerate(allowed_starts[a]):
            for j in range(len(offers[a].get_profile())):
                t = int((s - sim_start_ts) / res) + j
                key = (a, s_idx, t)
                if key in p:
                    val = pulp.value(p[key])
                    if val > 0:
                        sol["p"][a][t] = val
    for (a, t), var in pr_up.items(): sol["pr_up"][a][t] = value(var)
    for (a, t), var in pr_dn.items(): sol["pr_dn"][a][t] = value(var)
    for (a, t), var in pb_up.items(): sol["pb_up"][a][t] = value(var)
    for (a, t), var in pb_dn.items(): sol["pb_dn"][a][t] = value(var)
    for (a, t), var in s_up.items(): sol["s_up"][a][t] = value(var)
    for (a, t), var in s_dn.items(): sol["s_dn"][a][t] = value(var)

    # Compute and return profit as well
    return sol

def compute_profit(sol, spot, reserve, activation, indicators, penalty_series=None):
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
                r_up, r_dn, _, _ = reserve.iloc[t]
                res_rev += (pr_up_val * r_up + pr_dn_val * r_dn) * dt

            # activation revenue & penalty
            if config.RUN_ACTIVATION and activation is not None:
                pb_up_val = sol["pb_up"][a].get(t, 0.0) or 0.0
                pb_dn_val = sol["pb_dn"][a].get(t, 0.0) or 0.0
                b_up, b_dn, penalty, _, _ = activation.iloc[t]
                act_rev += (pb_up_val * b_up + pb_dn_val * b_dn) * dt

                s_up_val = sol["s_up"][a].get(t, 0.0)
                s_dn_val = sol["s_dn"][a].get(t, 0.0)
                penalty = penalty_series.iloc[t] if penalty_series is not None else config.PENALTY
                pen_cost += penalty * (s_up_val + s_dn_val) * dt

    total = res_rev + act_rev - spot_rev - pen_cost
    return {
        "spot_rev":  spot_rev,
        "res_rev":   res_rev,
        "act_rev":   act_rev,
        "penalty":   pen_cost,
        "total_rev": total
    }






