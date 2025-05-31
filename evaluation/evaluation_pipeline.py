# Entry point for evaluation
# Works with both FlexOffers and DFOs

from time import perf_counter
from evaluation.fleet_simulator import simulate_fleet
import json
from optimization.flexOfferOptimizer import BaseOptimizer
from typing import List, Dict, Any
from classes.electricVehicle import ElectricVehicle
import os
import numpy as np
import random
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from config import config
import time
import pandas as pd
from datetime import timedelta, datetime
from database.dataManager import load_and_prepare_prices
import matplotlib.pyplot as plt
from evaluation.utils.plot_flexOffer import plot_flexoffer, plot_flexoffer_aggregation
from evaluation.metrics import greedy_baseline_schedule, compute_profit, compute_financial_metrics, compute_mean_runtimes
from itertools import product
from helpers import sol_to_df, add_spot_prices_to_df
from flexoffer_logic import Flexoffer, DFO, TimeSlice, set_time_resolution

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_single_evaluation(flexoffers: List[Flexoffer], dfos: List[DFO],scenario: Dict[str, Any]) -> Dict[str, Any]:

    start_date = pd.to_datetime(config.SIMULATION_START_DATE)
    sim_start_ts = int(start_date.timestamp())
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day
    compute_optimal = config.NUM_EVS <= 100

    # --- Measure price‐loading time ---
    t0_price = perf_counter()

    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start_date,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )


    runtime_price_loading = perf_counter() - t0_price

    runtimes_agg, runtimes_sch = [], []
    daily_results = []

    for day in range(config.SIMULATION_DAYS):

        start_slot = day * slots_per_day
        end_slot = start_slot + slots_per_day

        active_dfos = []
        active_fos = []
        active = []

        for dfo in dfos:
            offset = int((dfo.get_est() - sim_start_ts) / config.TIME_RESOLUTION)
            dur = dfo.get_duration()
            if offset < end_slot and (offset + dur) > start_slot:
                active_dfos.append(dfo)

        for fo in flexoffers:
            offset = int((fo.get_est() - sim_start_ts) / config.TIME_RESOLUTION)
            dur = fo.get_duration()
            if offset < end_slot and (offset + dur) > start_slot:
                active_fos.append(fo)

        active = flexoffers if config.TYPE == "FO" else dfos

        # if len(active) < 2:
        #     continue

        t0 = time.time()
        agg_offers = cluster_and_aggregate_flexoffers(active, config.NUM_CLUSTERS)

        print(f"offer length {len(agg_offers)}")
        for offer in agg_offers:
            print(offer)

        optimizer_opt = BaseOptimizer(active_dfos, spot, reserve, activation, indicators)
        optimizer_agg = BaseOptimizer(agg_offers, spot, reserve, activation, indicators)

        clustering_time = time.time() - t0
        solution = optimizer_agg.run()
        scheduling_time = time.time() - t0 - clustering_time

        runtimes_agg.append(clustering_time)
        runtimes_sch.append(scheduling_time)

        greedy_solution = greedy_baseline_schedule(active_fos, horizon_slots)
        # if compute_optimal:
        optimal_solution = optimizer_opt.run_theoretical_optimum()
        # else:
        #     optimal_solution = greedy_solution = greedy_baseline_schedule(active_fos, horizon_slots)

        rev_sched = compute_profit(solution, spot, reserve, activation, indicators, penalty_series=activation['ImbalancePriceDKK'])
        rev_base = compute_profit(greedy_solution, spot, reserve, activation, indicators)
        rev_opt = compute_profit(optimal_solution, spot, reserve, activation, indicators, penalty_series=activation['ImbalancePriceDKK'])

        # active_fos[0].print_flexoffer()
        # active_fos[1].print_flexoffer()
        # agg_offers[0].print_flexoffer()

        # plot_flexoffer_aggregation(active_fos[0], active_fos[1], agg_offers[0], spot_prices=spot, resolution_seconds=config.TIME_RESOLUTION)

        # print(rev_sched)
        # print(rev_base)
        # print(rev_opt)

        daily_results.append({
            "rev_sched": rev_sched,
            "rev_base": rev_base,
            "rev_opt": rev_opt,
        })
    
    runtimes = compute_mean_runtimes(runtimes_sch, runtimes_agg)
    financials = compute_financial_metrics(daily_results)

    return {
        "scenario": scenario,
        # new timings:
        "runtime_price_loading": runtime_price_loading,
        **runtimes,
        **financials
    }



def evaluate_configurations():

    out_dir = 'evaluation/results'
    os.makedirs(out_dir, exist_ok=True)

    scenarios = get_scenarios()

    results = []
    for scenario in scenarios:
        np.random.seed(42)
        random.seed(42)
        # Override config with current scenario
    
        config.apply_override(scenario)

        # --- Measure simulate_fleet time ---
        t0_sim = perf_counter()
        fos, dfos = simulate_fleet(
            num_evs=config.NUM_EVS,
            start_date=pd.to_datetime(config.SIMULATION_START_DATE),
            simulation_days=config.SIMULATION_DAYS
        )
        runtime_simulation = perf_counter() - t0_sim

        res = run_single_evaluation(fos, dfos, scenario)
        res["runtime_simulation"] = runtime_simulation
        results.append(res)

    # Save CSV summary
    df = pd.DataFrame([
        {
            "TYPE": r["scenario"]["TYPE"],
            "MODE": r["scenario"]["MODE"],
            "RUN_SPOT": r["scenario"]["RUN_SPOT"],
            "RUN_RESERVE": r["scenario"]["RUN_RESERVE"],
            "RUN_ACTIVATION": r["scenario"]["RUN_ACTIVATION"],
            "NUM_EVS": r["scenario"]["NUM_EVS"],
            "TIME_RES": r["scenario"]["TIME_RESOLUTION"],
            "NUM_CLUSTERS": r["scenario"]["NUM_CLUSTERS"],
            "ALIGNMENT": r["scenario"]["ALIGNMENT"],
            "CLUSTER_METHOD": r["scenario"]["CLUSTER_METHOD"],
            "DYNAMIC_CLUSTERING": r["scenario"]["DYNAMIC_CLUSTERING"],
            "PARALLEL_CLUSTER_AGGREGATION": r["scenario"]["PARALLEL_CLUSTER_AGGREGATION"],
            "runtime_simulation": r["runtime_simulation"],
            "runtime_price_loading": r["runtime_price_loading"],
            "runtime_scheduling": r["runtime_scheduling"],
            "runtime_aggregation": r["runtime_aggregation"],
            # percent savings/gains
            "pct_total_saved":  r["pct_total_saved"],
            "pct_saved_spot":   r["pct_saved_spot"],
            "pct_gain_res":     r["pct_gain_res"],
            "pct_gain_act":     r["pct_gain_act"],
            # totals
            "pct_of_optimal": r["pct_of_optimal"],
        }
        for r in results
    ])
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)


def get_scenarios():
    types = ["FO"]
    modes = ["joint"]
    alignments = ["balance_fast"]
    run_spot_options = [True]
    run_reserve_options = [False]
    run_activation_options = [False]
    time_resolutions = [3600]
    cluster_methods = ['dbscan']
    dynamic = [False]
    parallel = [False]
    clusters = [5]
    num_evs = [1000]

    scenarios = []
    for type, mode, spot, reserve, activation, res, evs, cluster, align, cluster_method, dyn, par in product(
        types, modes, run_spot_options, run_reserve_options, run_activation_options, time_resolutions, num_evs, clusters, alignments, cluster_methods, dynamic, parallel
    ):
        # Skip invalid configs: activation can't be true if reserve is false
        if (not spot) or (reserve != activation) or (mode == 'sequential_reserve_first' and reserve == False):
            continue

        #Each scenario will overwrite the config with these settings
        scenarios.append({
            "TYPE": type,
            "MODE": mode,
            "RUN_SPOT": spot,
            "RUN_RESERVE": reserve,
            "NUM_CLUSTERS": cluster,
            "RUN_ACTIVATION": activation,
            "CLUSTER_METHOD": cluster_method,
            "DYNAMIC_CLUSTERING": dyn, 
            "PARALLEL_CLUSTER_AGGREGATION": par,
            "TIME_RESOLUTION": res,
            "NUM_EVS": evs,
            "ALIGNMENT": align,
        })
    return scenarios