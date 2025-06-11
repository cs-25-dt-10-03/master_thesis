# Entry point for evaluation
# Works with both FlexOffers and DFOs

from time import perf_counter
from evaluation.fleet_simulator import simulate_fleet
from optimization.flexOfferOptimizer import BaseOptimizer
from typing import List, Dict, Any, Tuple
import os
import numpy as np
import random
from cluster import cluster_and_aggregate_offers
from config import config
import pandas as pd
from datetime import timedelta, datetime
from database.dataManager import load_and_prepare_prices
from evaluation.utils.plot_flexOffer import plot_flexoffer, plot_flexoffer_aggregation
from evaluation.metrics import greedy_baseline_schedule, compute_profit, compute_financial_metrics, compute_mean_runtimes
from itertools import product
from helpers import slice_prices, filter_day_offers
from flexoffer_logic import Flexoffer, DFO, TimeSlice, set_time_resolution

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1) run_day_optimizations
def run_day_optimizations(
    flexoffers_day:  List[Flexoffer],
    dfos_day:        List[DFO],
    prices:          Dict[str, Any],
    start_slot:      int,
    end_slot:        int,
    slots_per_day:   int,
    compute_optimal: bool,
    sim_start_ts:    float
) -> Dict[str, Any]:
    """
    Slice prices, rebase to this day's start, cluster, run aggregator,
    greedy, and optimal, then compute profits.
    """
    # Rebase timestamp for slot‐0 of this day
    base_ts = sim_start_ts + start_slot * config.TIME_RESOLUTION

    # 1) slice prices for this 24h block
    spot_day, reserve_day, activation_day, indic_day, imb_day = slice_prices(prices, start_slot, end_slot)

    print(f"start_slot={start_slot}, end_slot={end_slot}, ",
        f"spot spans {spot_day.index[0]} → {spot_day.index[-1]}")

    # 2) cluster & aggregate
    if config.TYPE == "FO":
        agg_offers, t_clust, t_agg = cluster_and_aggregate_offers(flexoffers_day)
    else:
        agg_offers, t_clust, t_agg = cluster_and_aggregate_offers(dfos_day)
 
    # 3) aggregator solve (with Day-0 = base_ts)
    t0 = perf_counter()
    sol_agg = BaseOptimizer(agg_offers, spot_day, reserve_day ,activation_day, indic_day, base_ts=base_ts).run()
    t_sch = perf_counter() - t0

    rev_sched = compute_profit(sol_agg,spot_day,reserve_day,activation_day,indic_day,penalty_series=imb_day)

    # 4) greedy baseline
    sol_gr = greedy_baseline_schedule(flexoffers_day, slots_per_day, base_ts=base_ts)
    rev_base = compute_profit(sol_gr, spot_day, reserve_day, activation_day, indic_day)

    # 5) theoretical optimum
    if compute_optimal:
        # always run “optimal” on the raw DFOs
        sol_opt = BaseOptimizer(dfos_day, spot_day, reserve_day, activation_day, indic_day, base_ts=base_ts).run_theoretical_optimum()
    else:
        sol_opt = sol_gr
    rev_opt = compute_profit( sol_opt, spot_day, reserve_day, activation_day, indic_day, penalty_series=imb_day)

    return {
        "rev_sched":        rev_sched,
        "rev_base":         rev_base,
        "rev_opt":          rev_opt,
        "clustering_time":  t_clust,
        "aggregation_time": t_agg,
        "scheduling_time":  t_sch
    }

# 2) run_monthly_dayahead
def run_monthly(flexoffers: List[Any], dfos: List[Any], prices: Dict[str, Any], sim_start_ts: float) -> List[Dict[str, Any]]:
    """
    Loop over calendar days, filter offers, and delegate to run_day_optimizations.
    """
    slots_per_day   = int(24 * (3600 / config.TIME_RESOLUTION))
    compute_optimal = config.NUM_EVS <= 1001 # we only compute optimal if we have 1000 or less evs

    daily_results: List[Dict[str, Any]] = []
    for day in range(config.SIMULATION_DAYS):
    
        fos_day, dfos_day, start_slot, end_slot = filter_day_offers(flexoffers, dfos, sim_start_ts, day, slots_per_day)

        dr = run_day_optimizations(
            fos_day,
            dfos_day,
            prices,
            start_slot,
            end_slot,
            slots_per_day,
            compute_optimal,
            sim_start_ts
        )
        daily_results.append(dr)

    return daily_results


# 3) run_evaluation
def run_evaluation(
    flexoffers: List[Any],
    dfos:       List[Any],
    scenario:   Dict[str, Any]
) -> Dict[str, Any]:
    """
    Entry point: load prices once, run monthly DA, then summarize.
    """
    # 1) setup
    start_date   = pd.to_datetime(config.SIMULATION_START_DATE)
    sim_start_ts = float(start_date.timestamp())
    
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = (config.SIMULATION_DAYS + 1) * slots_per_day

    compute_optimal = config.NUM_EVS <= 1001

    # 2) load *all* markets
    t0 = perf_counter()
    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts       = start_date,
        horizon_slots  = horizon_slots,
        resolution     = config.TIME_RESOLUTION
    )
    runtime_price_loading = perf_counter() - t0

    prices = {
        "spot":       spot,
        "reserve":    reserve,
        "activation": activation,
        "indicators": indicators,
        "imbalance":  activation["ImbalancePriceDKK"]
    }

    # 3) rolling day‐ahead loop for a month
    daily_results = run_monthly(
        flexoffers,
        dfos,
        prices,
        sim_start_ts
    )

    # 4) collect mean runtimes for daily results
    runtimes_sch = [r["scheduling_time"]  for r in daily_results]
    runtimes_agg = [r["aggregation_time"] for r in daily_results]
    runtimes_cl  = [r["clustering_time"]  for r in daily_results]
    runtimes     = compute_mean_runtimes(runtimes_sch, runtimes_agg, runtimes_cl)

    # 5) financial metric
    financials = compute_financial_metrics(daily_results)

    return {
        "scenario":               scenario,
        "runtime_price_loading":  runtime_price_loading,
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
        fos, dfos = simulate_fleet(num_evs=config.NUM_EVS, start_date=pd.to_datetime(config.SIMULATION_START_DATE), simulation_days=config.SIMULATION_DAYS)
        runtime_simulation = perf_counter() - t0_sim

        res = run_evaluation(fos, dfos, scenario)
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
            "NUM_CLUSTERS": config.NUM_CLUSTERS,
            "ALIGNMENT": r["scenario"]["ALIGNMENT"],
            "CLUSTER_METHOD": r["scenario"]["CLUSTER_METHOD"],
            "DYNAMIC_CLUSTERING": r["scenario"]["DYNAMIC_CLUSTERING"],
            "PARALLEL_CLUSTER_AGGREGATION": r["scenario"]["PARALLEL_CLUSTER_AGGREGATION"],
            "runtime_simulation": r["runtime_simulation"],
            "runtime_price_loading": r["runtime_price_loading"],
            "runtime_scheduling": r["runtime_scheduling"],
            "runtime_aggregation": r["runtime_aggregation"],
            "runtime_clustering": r["runtime_clustering"],
            # percent savings/gains
            "pct_saved_vs_greedy_baseline":  r["pct_total_saved"],
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
    run_reserve_options = [True]
    run_activation_options = [True]
    time_resolutions = [3600]
    cluster_methods = ['ward']
    dynamic = [False]
    parallel = [False]
    clusters =  [15]
    num_evs = [15000]

    scenarios = []
    for type, mode, spot, reserve, activation, res, evs, cluster, align, cluster_method, dyn, par in product(
        types, modes, run_spot_options, run_reserve_options, run_activation_options, time_resolutions, num_evs, clusters, alignments, cluster_methods, dynamic, parallel
    ):
        # Skip invalid configs: activation can't be true if reserve is false
        if (not spot) or (reserve != activation) or (mode == 'sequential_reserve_first' and reserve == False) or (type == 'DFO' and align == 'balance') or (type == 'DFO' and align == 'balance_fast'):
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