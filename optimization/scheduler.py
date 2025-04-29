
from config import config
from optimization.flexOfferOptimizer import optimize_flexoffers

def schedule_offers(offers, spot_prices, reserve_prices, activation_prices, indicators):
    """
    Schedules FlexOffers according to current config settings.
    Returns: solution dict
    """

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

        # Lock p from spot scheduling
        locked_p = sol_spot["p"]

        # Step 2: Optimize reserve/activation separately, given spot schedule
        sol_reserve_activation = optimize_flexoffers(
            offers,
            spot_prices=[0]*len(spot_prices),  # no spot cost second pass
            reserve_prices=reserve_prices if config.RUN_RESERVE else None,
            activation_prices=activation_prices if config.RUN_ACTIVATION else None,
            indicators=indicators if config.RUN_ACTIVATION else None
        )

        # Merge solutions: p from spot; reserve/activation from second optimization
        merged_sol = sol_spot.copy()
        for key in ["pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn"]:
            if key in sol_reserve_activation:
                merged_sol[key] = sol_reserve_activation[key]

        return merged_sol

    else:
        raise ValueError(f"Unknown scheduling MODE: {config.MODE}")
