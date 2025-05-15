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
from optimization.scheduler import schedule_offers, schedule_offers_mpc_for_day
from evaluation.metrics import optimize_full_soc, greedy_baseline_schedule, compute_profit, compute_financial_metrics, compute_mean_runtimes
from itertools import product
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

        if len(active) < 2:
            continue

        print(f"length of active flexOffers {len(active)}")

        solution, clustering_time, scheduling_time = schedule_offers_mpc_for_day(active, day, spot, reserve, activation, indicators, mpc_horizon_days=1, skip_filter=True)
        runtimes_agg.append(clustering_time)
        runtimes_sch.append(scheduling_time)


        greedy_solution = greedy_baseline_schedule(active, horizon_slots)
        optimal_solution = schedule_offers(active, spot, reserve, activation, indicators)
        
        rev_sched = compute_profit(solution, spot, reserve, activation, indicators)
        rev_base = compute_profit(greedy_solution, spot, reserve, activation, indicators)
        rev_opt = compute_profit(optimal_solution, spot, reserve, activation, indicators)

        daily_results.append({
            "rev_sched": rev_sched,
            "rev_base": rev_base,
            "rev_opt": rev_opt,
        })
    
    runtimes = compute_mean_runtimes(runtimes_sch, runtimes_agg)
    financials = compute_financial_metrics(daily_results)

    return {
        "scenario": scenario, 
        "used_config": {
        k: getattr(config, k)
            for k in ["TIME_RESOLUTION", "NUM_EVS", "CLUSTER_METHOD", "NUM_CLUSTERS", "PENALTY", "MODE", "RUN_SPOT", "RUN_RESERVE", "RUN_ACTIVATION", "SIMULATION_DAYS"]
        },
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
            "MODE": r["scenario"]["MODE"],
            "RUN_SPOT": r["scenario"]["RUN_SPOT"],
            "RUN_RESERVE": r["scenario"]["RUN_RESERVE"],
            "RUN_ACTIVATION": r["scenario"]["RUN_ACTIVATION"],
            "NUM_EVS": r["scenario"]["NUM_EVS"],
            "TIME_RES": r["scenario"]["TIME_RESOLUTION"],
            "CLUSTER_METHOD": r["used_config"]["CLUSTER_METHOD"],
            "NUM_CLUSTERS": r["scenario"]["NUM_CLUSTERS"],
            "ALIGNMENT": r["scenario"]["ALIGNMENT"],
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
    modes = ["joint"]
    alignments = ["start"]
    run_spot_options = [True]
    run_reserve_options = [False]
    run_activation_options = [False]
    time_resolutions = [900]
    clusters = [5]
    num_evs = [30]

    scenarios = []
    for mode, spot, reserve, activation, res, evs, cluster, align in product(
        modes, run_spot_options, run_reserve_options, run_activation_options, time_resolutions, num_evs, clusters, alignments
    ):
        # Skip invalid configs: activation can't be true if reserve is false
        if activation and not reserve:
            continue

        #Each scenario will overwrite the config with these settings
        scenarios.append({
            "MODE": mode,
            "RUN_SPOT": spot,
            "RUN_RESERVE": reserve,
            "NUM_CLUSTERS": cluster,
            "RUN_ACTIVATION": activation,
            "TIME_RESOLUTION": res,
            "NUM_EVS": evs,
            "ALIGNMENT": align,
        })
    return scenarios