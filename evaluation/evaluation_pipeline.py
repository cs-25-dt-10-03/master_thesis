# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.fleet_simulator import simulate_fleet
import json
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
from optimization.scheduler import schedule_offers
from evaluation.metrics import greedy_baseline_schedule, compute_profit, compute_financial_metrics, compute_mean_runtimes, theoretical_optimal_schedule
from itertools import product
from helpers import sol_to_df, add_spot_prices_to_df
from flexoffer_logic import Flexoffer, DFO, TimeSlice, set_time_resolution

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_single_evaluation(flexoffers: List[Flexoffer], dfos: List[DFO],scenario: Dict[str, Any]) -> Dict[str, Any]:

    start_date = pd.to_datetime(config.SIMULATION_START_DATE)
    sim_start_ts = datetime.timestamp(start_date)
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start_date,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )

    runtimes_agg, runtimes_sch = [], []
    daily_results = []

    for day in range(config.SIMULATION_DAYS):

        start_slot = day * slots_per_day
        end_slot = start_slot + slots_per_day

        pool = flexoffers if config.TYPE == "FO" else dfos

        active = []
        active_fos = []

        for fo in pool:
            offset = int((fo.get_est() - sim_start_ts) / config.TIME_RESOLUTION)
            dur = fo.get_duration()
            if offset < end_slot and (offset + dur) > start_slot:
                active.append(fo)


        for fo in flexoffers:
            offset = int((fo.get_est() - sim_start_ts) / config.TIME_RESOLUTION)
            dur = fo.get_duration()
            if offset < end_slot and (offset + dur) > start_slot:
                active_fos.append(fo)

        if len(active) < 2:
            continue

        t0 = time.time()
        agg_offers = cluster_and_aggregate_flexoffers(active, config.NUM_CLUSTERS)
        clustering_time = time.time() - t0

        solution = schedule_offers(agg_offers, spot, reserve, activation, indicators)
        scheduling_time = time.time() - t0 - clustering_time
        solution_ind = schedule_offers(active_fos, spot, reserve, activation, indicators)

        runtimes_agg.append(clustering_time)
        runtimes_sch.append(scheduling_time)

        greedy_solution = greedy_baseline_schedule(active_fos, horizon_slots)
        optimal_solution = theoretical_optimal_schedule(active_fos, spot, reserve, activation, indicators)
        optimal_solution_agg = theoretical_optimal_schedule(agg_offers, spot, reserve, activation, indicators)

        for fo in active:
            fo.print_flexoffer()
        agg_offers[0].print_flexoffer()
    
        ind_rev_sched = compute_profit(solution_ind, spot, reserve, activation, indicators)
        rev_opt_agg = compute_profit(optimal_solution_agg, spot, reserve, activation, indicators, penalty_series=activation['ImbalancePriceDKK'])

        rev_sched = compute_profit(solution, spot, reserve, activation, indicators)
        rev_base = compute_profit(greedy_solution, spot, reserve, activation, indicators)
        rev_opt = compute_profit(optimal_solution, spot, reserve, activation, indicators, penalty_series=activation['ImbalancePriceDKK'])
 
        print(f"total revenue when we aggregate: {rev_sched['total_rev']}")
        print(f"total revenue for non-aggregated: {ind_rev_sched['total_rev']}")
        print(f"total theo revenue when we aggregate: {rev_opt_agg['total_rev']}")
        print(f"total theo revenue for non-aggregated: {rev_opt['total_rev']}")


        for fo in active_fos:
            energy = sum(fo.get_scheduled_allocation())
            if energy < 1e-3:
                print(f"[UNSCHEDULED] FO {fo.get_offer_id()} → {energy:.2f} kWh")


        # plot_schedule(flexoffers, spot, label="Individual")
        # plot_schedule(agg_offers, spot, label="Aggregated")
        # plt.legend(); plt.title("Scheduled Power vs. Spot Prices"); plt.show()

        print(rev_base)
        print(rev_opt)

        # df_sched = sol_to_df(solution)
        # df_theo  = sol_to_df(optimal_solution)

        # df_sched = add_spot_prices_to_df(df_sched, spot)
        # df_theo  = add_spot_prices_to_df(df_theo, spot)

        # # Save
        # os.makedirs("schedule", exist_ok=True)
        # df_sched.to_csv("schedule/scheduler_solution.csv", index=False)
        # df_theo.to_csv("schedule/theoretical_solution.csv", index=False)

        daily_results.append({
            "rev_sched": rev_sched,
            "rev_base": rev_base,
            "rev_opt": rev_opt,
        })
    
    runtimes = compute_mean_runtimes(runtimes_sch, runtimes_agg)
    financials = compute_financial_metrics(daily_results)

    return {
        "scenario": scenario, 
        **runtimes,
        **financials
    }


def plot_schedule(offers, spot, label):
    ts = []
    val = []
    for fo in offers:
        est = fo.get_scheduled_start_time()
        alloc = fo.get_scheduled_allocation()
        for i, a in enumerate(alloc):
            t = pd.to_datetime(est + i * config.TIME_RESOLUTION, unit="s")
            ts.append(t)
            val.append(a)
    plt.plot(ts, val, label=label)


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
        set_time_resolution(config.TIME_RESOLUTION)

        fos, dfos = simulate_fleet(
            num_evs=config.NUM_EVS,
            start_date=pd.to_datetime(config.SIMULATION_START_DATE),
            simulation_days=config.SIMULATION_DAYS
        )
        res = run_single_evaluation(fos, dfos, scenario)
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
    alignments = ["start"]
    run_spot_options = [True]
    run_reserve_options = [False]
    run_activation_options = [False]
    time_resolutions = [3600]
    cluster_methods = ['ward']
    dynamic = [False]
    parallel = [False]
    clusters = [1]
    num_evs = [2]

    scenarios = []
    for type, mode, spot, reserve, activation, res, evs, cluster, align, cluster_method, dyn, par in product(
        types, modes, run_spot_options, run_reserve_options, run_activation_options, time_resolutions, num_evs, clusters, alignments, cluster_methods, dynamic, parallel
    ):
        # Skip invalid configs: activation can't be true if reserve is false
        if activation and not reserve:
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