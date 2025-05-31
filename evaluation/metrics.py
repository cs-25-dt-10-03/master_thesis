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
        "runtime_scheduling": round(np.mean([runtimes_sch]), 3),
        "runtime_aggregation": round(np.mean([runtimes_agg]), 3)
    }

def compute_financial_metrics(daily_results):
    import numpy as np

    # --- 1) Pull out mean revenues from your daily results ---
    rev_sch_spot   = np.mean([r["rev_sched"]["spot_rev"] for r in daily_results])
    rev_sch_res    = np.mean([r["rev_sched"]["res_rev"]  for r in daily_results])
    rev_sch_act    = np.mean([r["rev_sched"]["act_rev"]  for r in daily_results])

    rev_opt_spot   = np.mean([r["rev_opt"]["spot_rev"]   for r in daily_results])
    rev_opt_res    = np.mean([r["rev_opt"]["res_rev"]    for r in daily_results])
    rev_opt_act    = np.mean([r["rev_opt"]["act_rev"]    for r in daily_results])

    rev_base_spot  = np.mean([r["rev_base"]["spot_rev"]  for r in daily_results])

    # --- 2) Interpret as positive costs / revenues ---
    baseline_cost        = rev_base_spot

    scheduler_spot_cost  = rev_sch_spot
    scheduler_res_rev    =   rev_sch_res
    scheduler_act_rev    =   rev_sch_act

    optimal_spot_cost    = rev_opt_spot
    optimal_res_rev      =   rev_opt_res
    optimal_act_rev      =   rev_opt_act

    # --- 3) Net profits (revenues minus costs) ---
    total_rev   = np.mean([r["rev_sched"]["total_rev"] for r in daily_results])
    optimal_total = np.mean([r["rev_opt"]["total_rev"]   for r in daily_results])
    
    # --- 4) % of theoretical maximum improvement captured (improvement ratio) ---
    # Compute net costs so that lower cost = better:
    #  scheduler_cost = spot_cost - (reserve_rev + activation_rev)
    #  optimal_cost   = optimal_spot_cost - (optimal_res_rev + optimal_act_rev)
    scheduler_cost = scheduler_spot_cost - (scheduler_res_rev + scheduler_act_rev)
    optimal_cost   = optimal_spot_cost   - (optimal_res_rev   + optimal_act_rev)

    # The maximum possible savings = baseline_cost - optimal_cost
    max_savings = baseline_cost - optimal_cost
    # The savings we actually achieved
    savings     = baseline_cost - scheduler_cost
    if max_savings <= 1e-9:
        pct_of_optimal = None
    else:
        pct_of_optimal = 100.0 * savings / max_savings
        # clamp into [0,100]
        pct_of_optimal = max(0.0, min(100.0, pct_of_optimal))
    # --- 5) Compute net scheduler cost & baseline savings ---
    scheduler_cost  = scheduler_spot_cost - (scheduler_res_rev + scheduler_act_rev)
    total_savings   = baseline_cost - scheduler_cost

    # savings vs baseline

    print(f"[DEBUGGING] scheduler_cost: {scheduler_cost}, baseline_cost: {baseline_cost} , total_savings: {total_savings} ")

    pct_total_saved = 100.0 * total_savings / baseline_cost
    print(f"[DEBUGGING] {pct_total_saved} = 100.0 * {total_savings} / {baseline_cost} ")


    # --- 6) Breakdown of contributions (sum ≈ 100%) ---
    if abs(total_savings) < 1e-9:
        pct_saved_spot = pct_gain_res = pct_gain_act = None
    else:
        saved_spot     = baseline_cost - scheduler_spot_cost
        gain_res       = scheduler_res_rev
        gain_act       = scheduler_act_rev

        print(f"saved spot: {saved_spot} , gain_res: {gain_res} , gain_act: {gain_act}")

        pct_saved_spot = 100.0 * saved_spot / total_savings
        pct_gain_res   = 100.0 * gain_res   / total_savings
        pct_gain_act   = 100.0 * gain_act   / total_savings

        # sanity check
        total_pct = pct_saved_spot + pct_gain_res + pct_gain_act
        if abs(total_pct - 100.0) > 1e-6:
            print(f"[WARNING] contributions sum to {total_pct:.4f}%")

    return {
        "pct_of_optimal": round(pct_of_optimal, 2),
        "pct_total_saved": round(pct_total_saved, 2),
        "pct_saved_spot": round(pct_saved_spot, 2),
        "pct_gain_res": round(pct_gain_res, 2),
        "pct_gain_act": round(pct_gain_act, 2),

    }

def greedy_baseline_schedule(offers, horizon):
    """
    Greedy baseline: always use sim_start_ts to index into prices.
    """
    sol = {k: {} for k in ("p","pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn")}
    if not offers:
        return sol

    slot_sec = config.TIME_RESOLUTION     # seconds per slot
    dt       = slot_sec / 3600.0          # hours per slot
    # rebase epoch so slot 0 is at the earliest of configured start or the first FO
    config_start_ts = int(pd.to_datetime(config.SIMULATION_START_DATE).timestamp())
    min_offer_ts    = min(fo.get_est() for fo in offers)
    sim_start_ts    = min(config_start_ts, min_offer_ts)
    T = horizon

    for a, fo in enumerate(offers):
        # total energy requirement (kWh)
        required_energy = fo.get_min_overall_alloc()
        prof            = fo.get_profile()
        dur             = fo.get_duration()
        start_ts        = fo.get_est()
        slot_energy_limit = prof[0].max_power * dt

        # map earliest_start to a 0-based slot index
        base_idx = int((start_ts - sim_start_ts) / slot_sec)

        remaining = required_energy
        p_dict    = {}

        # --- clamp every scheduled slot to [0, T)
        # greedily fill slot by slot, respecting each slice’s max_power
        for i in range(dur):
            if remaining <= 0:
                break
            idx = base_idx + i
            if idx < 0 or idx >= T:
                print("vi breaker tids horizon i greedy")
                break

            # allocate kWh up to slot limit or what's left
            energy_to_charge = min(slot_energy_limit, remaining)  # kWh
            p_val = energy_to_charge / dt # kW
            p_dict[idx] = p_val
            remaining   -= energy_to_charge

        sol["p"][a] = p_dict
        for key in ("pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"):
            sol[key][a] = {}

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






