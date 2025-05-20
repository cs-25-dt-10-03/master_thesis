from database.dataManager import load_and_prepare_prices
from config import config
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from optimization.flexOfferOptimizer import optimize_flexoffers, optimize_offers
from optimization.test_optimizers import optimize_stochastic
from datetime import datetime, timedelta
import time
from datetime import timedelta
from typing import List, Dict
import pandas as pd
from flexoffer_logic import Flexoffer, DFO

def schedule_offers(offers, spot_prices=None, reserve_prices=None, activation_prices=None, indicators=None):
    """
    Schedules FlexOffers according to current config settings.
    Returns: solution dict
    """
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    if spot_prices is None:
        spot_prices, reserve_prices, activation_prices, indicators = load_and_prepare_prices(
            start_ts=config.SIMULATION_START_DATE,
            horizon_slots=horizon_slots,
            resolution=config.TIME_RESOLUTION
        )


    if config.MODE == "joint":
        # All markets optimized together

        sol = optimize_offers(
            offers,
            spot_prices,
            reserve_prices=reserve_prices if config.RUN_RESERVE else None,
            activation_prices=activation_prices if config.RUN_ACTIVATION else None,
            indicators=indicators if config.RUN_ACTIVATION else None
        )
        return sol

    elif config.MODE == "sequential":
        # Step 1: Optimize spot-only

        sol_spot = optimize_offers(
            offers,
            spot_prices,
            reserve_prices=None,
            activation_prices=None,
            indicators=None
        )

        # Step 2: Optimize reserve/activation separately, given spot schedule
        zero_spot = pd.Series(0.0, index=spot_prices.index)

        sol_reserve_activation = optimize_offers(
            offers,
            spot_prices=zero_spot,  # no spot cost second pass
            reserve_prices=reserve_prices if config.RUN_RESERVE else None,
            activation_prices=activation_prices if config.RUN_ACTIVATION else None,
            indicators=indicators if config.RUN_ACTIVATION else None,
            fixed_p = sol_spot['p']
        )

        # Merge solutions: p from spot; reserve/activation from second optimization
        merged_sol = sol_spot.copy()
        for key in ["pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn"]:
            if key in sol_reserve_activation:
                merged_sol[key] = sol_reserve_activation[key]

        return merged_sol
    

    elif config.MODE == "sequential_reserve_first":
        # Step 1: optimize reserve/activation only (ignore spot)
        zero_spot = pd.Series(0.0, index=spot_prices.index)

        sol_res = optimize_offers(
            offers,
            spot_prices=zero_spot,
            reserve_prices=reserve_prices if config.RUN_RESERVE else None,
            activation_prices=activation_prices if config.RUN_ACTIVATION else None,
            indicators=indicators if config.RUN_ACTIVATION else None
        )

        # Step 2: optimize spot only, with reserve allocations held fixed

        sol_spot = optimize_offers(
            offers,
            spot_prices=spot_prices,
            reserve_prices=None,
            activation_prices=None,
            indicators=None,
            fixed_pr_up=sol_res["pr_up"],
            fixed_pr_dn=sol_res["pr_dn"]
        )


        # Merge: keep reserves from sol_res, spot p from sol_spot
        merged = sol_res.copy()
        merged["p"] = sol_spot["p"]
        return merged

    else:
        raise ValueError(f"Unknown scheduling MODE: {config.MODE}")

