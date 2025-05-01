from database.dataManager import load_and_prepare_prices
from config import config
from optimization.flexOfferOptimizer import optimize_flexoffers


from datetime import timedelta
from typing import List, Dict
import pandas as pd
from flexoffer_logic import Flexoffer

def schedule_offers(offers):
    """
    Schedules FlexOffers according to current config settings.
    Returns: solution dict
    """
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    print(f"START TID: {config.SIMULATION_START_DATE} \n")
    print(f"slots per dag: {slots_per_day} \n")
    print(f"Hvor mange slots i alt: {horizon_slots} \n")



    spot_prices, reserve_prices, activation_prices, indicators = load_and_prepare_prices(
        start_ts=config.SIMULATION_START_DATE,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )


    if config.MODE == "joint":
        # All markets optimized together
        sol = optimize_flexoffers(
            offers,
            spot_prices,
            reserve_prices=reserve_prices if config.RUN_RESERVE else None,
            activation_prices=activation_prices if config.RUN_ACTIVATION else None,
            indicators=indicators if config.RUN_ACTIVATION else None
        )
        return sol

    elif config.MODE == "sequential":
        # Step 1: Optimize spot-only
        sol_spot = optimize_flexoffers(
            offers,
            spot_prices,
            reserve_prices=None,
            activation_prices=None,
            indicators=None
        )

        # Step 2: Optimize reserve/activation separately, given spot schedule
        zero_spot = pd.Series(0.0, index=spot_prices.index)
        sol_reserve_activation = optimize_flexoffers(
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

    else:
        raise ValueError(f"Unknown scheduling MODE: {config.MODE}")
