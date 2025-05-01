# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.utils.plots import plot_results
from evaluation.fleet_simulator import simulate_fleet
import json
from classes.electricVehicle import ElectricVehicle
import os
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers, aggregate_rolling_horizon
from config import config
import time
import pandas as pd
from datetime import timedelta, datetime
from optimization.scheduler import schedule_offers

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_single_evaluation(spot, reserve, activation, indicators, scenario):
    
    # Override config with current scenario
    config.apply_override(scenario)

    start_date=pd.to_datetime(config.SIMULATION_START_DATE)
    current=pd.to_datetime(config.SIMULATION_START_DATE)

    simulation_days=config.SIMULATION_DAYS

    # 1: simulate fleet
    offers = simulate_fleet(
        num_evs=config.NUM_EVS,
        start_date=start_date,
        simulation_days=simulation_days
    )
    # 2: aggregate offers in a rolling window (we look at 24 hours at a time)
    end_date = start_date + timedelta(days=config.SIMULATION_DAYS)
    # we save the runtimes for each day and find the avg
    runtime_sch = []
    runtime_agg = []

    while current < end_date:

        current_ts = datetime.timestamp(current)
        end_date_ts = datetime.timestamp(end_date)

        window_h = 24
        active = [fo for fo in offers if fo.get_est() >= current_ts and fo.get_est() < end_date_ts]  

        if not active:
            return []
        
        print(type(active[0]))

        start_agg = time.time()
        agg_offers  = cluster_and_aggregate_flexoffers(active)
        runtime_agg.append(time.time() - start_agg)

        start_sch = time.time()
        solution = schedule_offers(agg_offers)
        runtime_sch.append(time.time() - start_sch)

        current += timedelta(hours = window_h)
        
    mean_runtime_scheduling = sum(runtime_sch) / len(runtime_sch)
    mean_runtime_aggregation = sum(runtime_agg) / len(runtime_agg)

    # 4: Compute profits and savings
    revs = compute_profit(solution, spot, reserve, activation, indicators)

    # 5. Export each scheduled allocation + start time
    schedules = [{
        "offer": a,
        "start_time": offers[a].get_scheduled_start_time(),
        "allocation": offers[a].get_scheduled_allocation()
    } for a in range(len(offers))]

    return {
        "scenario": scenario, 
        "runtime_aggregation": mean_runtime_aggregation,
        "runtime_scheduling": mean_runtime_scheduling,
        "schedules": schedules,
        "used_config": {
        k: getattr(config, k)
            for k in ["TIME_RESOLUTION", "NUM_EVS", "PENALTY", "MODE", "RUN_SPOT", "RUN_RESERVE", "RUN_ACTIVATION", "SIMULATION_DAYS"]
        },
        **revs
    }


def evaluate_configurations(spot, reserve, activation, indicators):

    out_dir = 'evaluation/results'
    os.makedirs(out_dir, exist_ok=True)
    scenarios = get_scenarios()

    results = []

    for scenario in scenarios:
        res = run_single_evaluation(spot, reserve, activation, indicators, scenario)
        results.append(res)

    # Save CSV summary
    df = pd.DataFrame([
        {
            "MODE": r["scenario"]["MODE"],
            "RUN_SPOT": r["scenario"]["RUN_SPOT"],
            "RUN_RESERVE": r["scenario"]["RUN_RESERVE"],
            "RUN_ACTIVATION": r["scenario"]["RUN_ACTIVATION"],
            "NUM_EVS": r["used_config"]["NUM_EVS"],
            "TIME_RES": r["used_config"]["TIME_RESOLUTION"],
            "runtime_scheduling": r["runtime_scheduling"],
            "runtime_aggregation": r["runtime_aggregation"],
            "total_rev": r["total_rev"],
            "spot_rev": r["spot_rev"],
            "res_rev": r["res_rev"],
            "act_rev": r["act_rev"],
            "penalty": r["penalty"]
        }
        for r in results
    ])
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def get_scenarios():
    return [
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False},
        #{"MODE": "sequential", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False, "NUM_EVS": 10},
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
            if config.RUN_RESERVE:
                pr_up_val = sol["pr_up"][a].get(t, 0.0)
                pr_dn_val = sol["pr_dn"][a].get(t, 0.0)
                r_up, r_dn = reserve.iloc[t]
                res_rev += (pr_up_val * r_up + pr_dn_val * r_dn) * dt

            # activation revenue & penalty
            if config.RUN_ACTIVATION:
                pb_up_val = sol["pb_up"][a].get(t, 0.0)
                pb_dn_val = sol["pb_dn"][a].get(t, 0.0)
                b_up, b_dn = activation.iloc[t]
                act_rev += (pb_up_val * b_up + pb_dn_val * b_dn) * dt

                s_up_val = sol["s_up"][a].get(t, 0.0)
                s_dn_val = sol["s_dn"][a].get(t, 0.0)
                pen_cost += config.PENALTY * (s_up_val + s_dn_val) * dt

    total = spot_rev + res_rev + act_rev - pen_cost
    return {
        "spot_rev":  spot_rev,
        "res_rev":   res_rev,
        "act_rev":   act_rev,
        "penalty":   pen_cost,
        "total_rev": total
    }
