# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.utils.plots import plot_results
import json
from classes.electricVehicle import ElectricVehicle
import os
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from config import config
import time
from optimization.flexOfferOptimizer import optimize

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_evaluation(ev_list, spot, reserve, activation, indicators, scenario):

    offers = [ev.create_synthetic_flexoffer(tec_fo=True) for ev in ev_list]
    # 1: schedule offers

    start = time.time()
    AFOs = cluster_and_aggregate_flexoffers(offers, config.NUM_CLUSTERS)
    runtime_aggregation = start - time.time()

    start = time.time()
    solution = optimize(AFOs)
    runtime_scheduling = start - time.time()

    # 2: Compute profits and savings
    profit = compute_profit(offers)
    baseline_cost = compute_baseline(offers)

    savings = (profit / baseline_cost) * 100

    # 4. Export each scheduled allocation + start time
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


def evaluate_configurations(configurations, spot, reserve, activation, indicators):

    ev_list = [
        ElectricVehicle(vehicle_id=i, capacity=100, soc_min=0.7, soc_max=0.9, charging_power=7.0, charging_efficiency=0.95)
        for i in range(config.NUM_EVS)
    ]

    results = []

    for scenario in configurations:
        res = run_single_evaluation(ev_list, spot, reserve, activation, indicators, scenario)
        results.append({
            "TIME_RESOLUTION": res["TIME_RESOLUTION"],
            "SIMULATION_DAYS": res["SIMULATION_DAYS"],
            "MODE": res["MODE"],
            "SPOT": res["RUN_SPOT"],
            "RESERVE": res["RUN_RESERVE"],
            "ACTIVATION": res["RUN_ACTIVATION"],
            "NUM_EVS": res["NUM_EVS"],
            "savings": res["savings"],
            "runtime": res["runtime"]})



def compute_profit(solution, spot, reserve, activation):
    #compute spot price
    spot_price = 0
    dt = config.TIME_RESOLUTION / 3600
    for p_val in solution["p"].values():
        for t, p in p_val.items():
            spot_price += spot[t] * p * dt

    # compute Reserve revenue (up and down)
    reserve_revenue = sum(
        reserve[t][0] * pr
        for pr_vals in solution["pr_up"].values()
        for t, pr in pr_vals.items()
    ) + sum(
        reserve[t][1] * pr
        for pr_vals in solution["pr_dn"].values()
        for t, pr in pr_vals.items()
    )

    # compute Activation revenue (up and down)
    activation_revenue = sum(
        activation[t][0] * pb
        for pb_vals in solution["pb_up"].values()
        for t, pb in pb_vals.items()
    ) + sum(
        activation[t][1] * pb
        for pb_vals in solution["pb_dn"].values()
        for t, pb in pb_vals.items()
    )

    # Penalty cost
    penalty_cost = config.PENALTY * (
        sum(s for s_vals in solution["s_up"].values() for s in s_vals.values()) +
        sum(s for s_vals in solution["s_dn"].values() for s in s_vals.values())
    )

    return spot_price + reserve_revenue + activation_revenue - penalty_cost



def compute_baseline(offers):
    """
    Computes the averages energy needed when buying energy as early as possible.
    offers: list of flexOffers
    """
    baseline = sum(fo.get_total_energy() for fo in offers)
    return baseline
