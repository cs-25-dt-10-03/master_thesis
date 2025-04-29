# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.utils.plots import plot_results
from evaluation.fleet_simulator import simulate_fleet
import json
from classes.electricVehicle import ElectricVehicle
import os
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from config import config
import time
import pandas as pd
from optimization.scheduler import schedule_offers

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_single_evaluation(spot, reserve, activation, indicators, scenario):
    
    # Override config with current scenario
    config.apply_override(scenario)


    # 1: simulate fleet
    offers = simulate_fleet(
        num_evs=config.NUM_EVS,
        start_date=config.SIMULATION_START_DATE,
        simulation_days=config.SIMULATION_DAYS
    )
    # 2: aggregate offers
    start = time.time()
    AFOs = cluster_and_aggregate_flexoffers(offers, config.NUM_CLUSTERS)
    runtime_aggregation = start - time.time()

    # 3: schedule offers
    start = time.time()
    solution = schedule_offers(AFOs, spot, reserve, activation, indicators)
    runtime_scheduling = start - time.time()

    # 4: Compute profits and savings
    profit = compute_profit(solution, spot, reserve, activation,)
    baseline_cost = compute_baseline(offers)
    savings = (profit / baseline_cost) * 100

    # 5. Export each scheduled allocation + start time
    schedules = [{
        "offer": a,
        "start_time": offers[a].get_scheduled_start_time(),
        "allocation": offers[a].get_scheduled_allocation()
    } for a in range(len(offers))]

    return {
        "scenario": scenario, 
        "savings": savings,
        "runtime_aggregation": runtime_aggregation,
        "runtime_scheduling": runtime_scheduling,
        "schedules": schedules,

        "used_config": {
        k: getattr(config, k)
            for k in ["TIME_RESOLUTION", "NUM_EVS", "PENALTY", "MODE", "RUN_SPOT", "RUN_RESERVE", "RUN_ACTIVATION", "SIMULATION_DAYS"]
        }
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
            "runtime_scheduling": r["runtime_scheduling"],
            "runtime_aggregation": r["runtime_aggregation"]
        }
        for r in results
    ])
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Save full detailed JSON
    with open(os.path.join(out_dir, "details.json"), "w") as f:
        json.dump(results, f, indent=2)


def get_scenarios():
    return [
        {"MODE": "joint", "RUN_SPOT": True, "RUN_RESERVE": False, "RUN_ACTIVATION": False},
    ]


def compute_profit(solution, spot_prices, reserve_prices, activation_prices):
    """
    solution: dict with keys "p","pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"
    spot_prices: pd.Series indexed by Timestamp (length T)
    reserve_prices: pd.DataFrame [Up,Down] shape (T,2)
    activation_prices: pd.DataFrame [Up,Down] shape (T,2)
    """
    dt = config.TIME_RESOLUTION / 3600.0

    # --- Spot cost (negative revenue) ---
    spot_rev = sum(
        spot_prices.iloc[t]  * p_val[t] * dt
        for p_val in solution["p"].values()
        for t in p_val
    )

    # --- Reserve revenue ---
    res_up   = sum(reserve_prices.iloc[t,0] * pr
                   for pr_vals in solution["pr_up"].values()
                   for t,pr in pr_vals.items())
    res_down = sum(reserve_prices.iloc[t,1] * pr
                   for pr_vals in solution["pr_dn"].values()
                   for t,pr in pr_vals.items())

    # --- Activation revenue ---
    act_up   = sum(activation_prices.iloc[t,0] * pb
                   for pb_vals in solution["pb_up"].values()
                   for t,pb in pb_vals.items())
    act_down = sum(activation_prices.iloc[t,1] * pb
                   for pb_vals in solution["pb_dn"].values()
                   for t,pb in pb_vals.items())

    # --- Penalties ---
    pen = config.PENALTY * (
        sum(s for s_vals in solution["s_up"].values() for s in s_vals.values()) +
        sum(s for s_vals in solution["s_dn"].values() for s in s_vals.values())
    )

    # *Note spot_rev is positive energy‐costs, so net profit = (reserve+activation) − spot_rev − pen 
    return (res_up + res_down + act_up + act_down) - spot_rev - pen



def compute_baseline(offers):
    """
    Computes the averages energy needed when buying energy as early as possible.
    offers: list of flexOffers
    """
    baseline = sum(fo.get_total_energy() for fo in offers)
    return baseline
