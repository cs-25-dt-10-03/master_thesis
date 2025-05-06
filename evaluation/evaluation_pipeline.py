# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.fleet_simulator import simulate_fleet
import json
from evaluation.metrics import schedule_dfos_theoretical
from classes.electricVehicle import ElectricVehicle
import os
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from config import config
import time
import pandas as pd
from datetime import timedelta, datetime
from database.dataManager import load_and_prepare_prices
from optimization.scheduler import schedule_offers
from flexoffer_logic import Flexoffer, DFO, TimeSlice

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_single_evaluation(fos, dfos, scenario):
    
    # Override config with current scenario
    config.apply_override(scenario)

    start = config.SIMULATION_START_DATE
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )
    print(f"længde på spot data i evaluation: {len(spot)}")

    start_date=pd.to_datetime(config.SIMULATION_START_DATE)
    current=pd.to_datetime(config.SIMULATION_START_DATE)

    # 2: aggregate offers in a rolling window (we look at 24 hours at a time)
    end_date = start_date + timedelta(days=config.SIMULATION_DAYS)

    # we save the runtimes for each day and find the avg
    runtime_sch = []
    runtime_agg = []
    days = []

    while current < end_date:

        current_ts = datetime.timestamp(current)

        window_h = 24
        active_fos = [fo for fo in fos if fo.get_est() >= current_ts and fo.get_est() < current_ts + window_h * 3600]  
        active_dfos = [dfo for dfo in dfos if dfo.get_est() >= current_ts and dfo.get_est() < current_ts + window_h * 3600]  

        if config.TYPE == 'FO':
            active = active_fos
        elif config.TYPE == 'DFO':
            active = active_dfos

        print(f"length of active flexOffer for this day: {len(active)}")

        if not active_fos:
            return []

        start_agg = time.time()
        agg_offers = cluster_and_aggregate_flexoffers(active, config.NUM_CLUSTERS)
        runtime_agg.append(time.time() - start_agg)

        print(f"length of aggregated flexoffers in the evaluation: {len(agg_offers)}")


        start_sch = time.time()
        solution = schedule_offers(agg_offers)
        runtime_sch.append(time.time() - start_sch)

        greedy_solution = greedy_baseline_schedule(active_fos)
        optimal_solution = schedule_offers(active_dfos)

        rev_sch = compute_profit(solution, spot, reserve, activation, indicators)
        sch_spot = rev_sch["spot_rev"]
        sch_res  = rev_sch["res_rev"]
        sch_act  = rev_sch["act_rev"]
        
        rev_baseline = compute_profit(greedy_solution, spot, reserve, activation, indicators)
        base_spot = rev_baseline["spot_rev"]
        base_res  = rev_baseline["res_rev"]   # should be zero (Laver kun spot market alloc i baseline)
        base_act  = rev_baseline["act_rev"]   # likewise usually zero   
        
        rev_optimal = compute_profit(optimal_solution, spot, reserve, activation, indicators)
        opt_total = rev_optimal["total_rev"]  

        days.append({
            "sch": rev_sch,
            "base": rev_baseline,
            "sch_spot_rev": sch_spot,
            "sch_res_rev":  sch_res,
            "sch_act_rev":  sch_act,
            "base_spot_rev": base_spot,
            "base_res_rev":  base_res,
            "base_act_rev":  base_act,
            "opt_total_rev": opt_total,
        })


        current += timedelta(hours = window_h)
        
    mean_runtime_scheduling = round(sum(runtime_sch) / len(runtime_sch), 3)
    mean_runtime_aggregation = round(sum(runtime_agg) / len(runtime_agg), 3)

    n = len(days)

    # mean per-market for scheduler
    mean_sch_spot = sum(r["sch_spot_rev"] for r in days) / n
    mean_sch_res  = sum(r["sch_res_rev"]  for r in days) / n
    mean_sch_act  = sum(r["sch_act_rev"]  for r in days) / n

    # mean per-market for baseline
    mean_base_spot = sum(r["base_spot_rev"] for r in days) / n
    mean_base_res  = sum(r["base_res_rev"]  for r in days) / n
    mean_base_act  = sum(r["base_act_rev"]  for r in days) / n

    # mean overall for theoretical optimum
    optimal_total = sum(r["opt_total_rev"] for r in days) / n


    # compute savings per market
    saved_spot = mean_base_spot - mean_sch_spot          # cost reduction
    gain_res   = mean_sch_res  - mean_base_res           # revenue gain
    gain_act   = mean_sch_act  - mean_base_act           # revenue gain

    # percent savings/gains relative to baseline spot cost
    # baseline_res and baseline_act are zero for greedy baseline
    pct_saved_spot = 100.0 * saved_spot / mean_base_spot if mean_base_spot else None
    pct_gain_res   = 100.0 * gain_res   / mean_base_spot if mean_base_spot else None
    pct_gain_act   = 100.0 * gain_act   / mean_base_spot if mean_base_spot else None

    # overall total
    total_rev = mean_sch_res + mean_sch_act - mean_sch_spot

    print(f"theo optimal: {optimal_total}")
    print(f"schedule result: {total_rev}")


    # overall percent compared to optimal
    pct_of_optimal = (optimal_total / total_rev) * 100 if optimal_total else None


    # overall percent saving relative to baseline cost
    pct_total_saved = 100.0 * (saved_spot + gain_res + gain_act) / mean_base_spot \
                      if mean_base_spot else None

    # 5. Export each scheduled allocation + start time
    schedules = [{
        "offer": a,
        "start_time": fos[a].get_scheduled_start_time(),
        "allocation": fos[a].get_scheduled_allocation()
    } for a in range(len(fos))]

    return {
        "scenario": scenario, 
        "runtime_aggregation": mean_runtime_aggregation,
        "runtime_scheduling": mean_runtime_scheduling,
        "schedules": schedules,
        "used_config": {
        k: getattr(config, k)
            for k in ["TIME_RESOLUTION", "NUM_EVS", "CLUSTER_METHOD", "NUM_CLUSTERS", "PENALTY", "MODE", "RUN_SPOT", "RUN_RESERVE", "RUN_ACTIVATION", "SIMULATION_DAYS"]
        },
        "mean_sch_spot":     round(mean_sch_spot,   3),
        "mean_sch_res":      round(mean_sch_res,    3),
        "mean_sch_act":      round(mean_sch_act,    3),
        "mean_base_spot":    round(mean_base_spot,  3),
        "mean_base_res":     round(mean_base_res,   3),
        "mean_base_act":     round(mean_base_act,   3),

        # absolute savings/gains
        "saved_spot":        round(saved_spot,      3),
        "gain_res":          round(gain_res,        3),
        "gain_act":          round(gain_act,        3),

        # percent savings/gains
        "pct_saved_spot":    round(pct_saved_spot,  2) if pct_saved_spot is not None else None,
        "pct_gain_res":      round(pct_gain_res,    2) if pct_gain_res is not None else None,
        "pct_gain_act":      round(pct_gain_act,    2) if pct_gain_act is not None else None,

        # overall
        "total_rev":         round(total_rev,       3),
        "pct_total_saved":   round(pct_total_saved, 2) if pct_total_saved is not None else None,

        # part of optimal
        "pct_of_optimal": round(pct_of_optimal, 3) if pct_of_optimal is not None else None
    }


def evaluate_configurations():

    out_dir = 'evaluation/results'
    os.makedirs(out_dir, exist_ok=True)
    scenarios = get_scenarios()

    start_date=pd.to_datetime(config.SIMULATION_START_DATE)
    simulation_days=config.SIMULATION_DAYS

    # 1: simulate fleet
    fos, dfos = simulate_fleet(
        num_evs=config.NUM_EVS,
        start_date=start_date,
        simulation_days=simulation_days
    )

    print(f"START TID: {config.SIMULATION_START_DATE} \n")

    results = []
    for scenario in scenarios:
        res = run_single_evaluation(fos, dfos, scenario)
        results.append(res)

    # Save CSV summary
    df = pd.DataFrame([
        {
            "MODE": r["scenario"]["MODE"],
            "RUN_SPOT": r["scenario"]["RUN_SPOT"],
            "RUN_RESERVE": r["scenario"]["RUN_RESERVE"],
            "RUN_ACTIVATION": r["scenario"]["RUN_ACTIVATION"],
            "NUM_EVS": r["scenario"]["NUM_EVS"],
            "TIME_RES": r["scenario"]["TIME_RESOLUTION"],
            "CLUSTER_METHOD": r["used_config"]["CLUSTER_METHOD"],
            "NUM_CLUSTERS": r["used_config"]["NUM_CLUSTERS"],
            "runtime_scheduling": r["runtime_scheduling"],
            "runtime_aggregation": r["runtime_aggregation"],
            # percent savings/gains
            "pct_saved_spot":   r["pct_saved_spot"],
            "pct_gain_res":     r["pct_gain_res"],
            "pct_gain_act":     r["pct_gain_act"],
            # totals
            "pct_total_saved":  r["pct_total_saved"],
            "total_rev":        r["total_rev"],
            "mean_base_spot": r["mean_base_spot"],
            "pct_of_optimal": r["pct_of_optimal"],
        }
        for r in results
    ])
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)



def get_scenarios():
    return [
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 20},

        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 20},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 20},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 20},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 20},

        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 100},

        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 100},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 100},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 100},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 100},

        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 3600, "NUM_EVS": 200},

        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": False, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
        {"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": True, "RUN_ACTIVATION": True, "TIME_RESOLUTION": 900, "NUM_EVS": 200},
    ]


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


def greedy_baseline_schedule(agg_offers):
    """
    Greedy baseline: always use sim_start_ts to index into prices.
    """
    sol = {k: {} for k in ("p","pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn")}
    if not agg_offers:
        return sol

    slot_sec = config.TIME_RESOLUTION  # seconds per slot
    dt       = slot_sec / 3600.0       # hours per slot
    sim_start_ts = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    T = 24 / dt

    for a, fo in enumerate(agg_offers):
        max_total = fo.get_max_overall_alloc()
        prof      = fo.get_profile()
        dur       = fo.get_duration()
        start_ts  = fo.get_est()
        max_pow   = prof[0].max_power

        base_idx  = int((start_ts - sim_start_ts) / slot_sec)

        remaining = max_total
        p_dict    = {}
        for i in range(dur):
            if remaining <= 0:
                break
            idx = base_idx + i
            if idx >= T:
                break
            energy_slot = min(max_pow * dt, remaining)
            p_val       = energy_slot / dt
            p_dict[idx] = p_val
            remaining  -= energy_slot

        sol["p"][a] = p_dict
        for key in ("pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"):
            sol[key][a] = {}

    return sol