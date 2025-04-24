import pandas as pd
import numpy as np
from optimization.simulator import create_aggregated_offers
from optimization.flexOfferOptimizer import optimize
from config import config

def compute_profit(solution, spot, reserve, activation):
    dt = config.TIME_RESOLUTION / 3600.0
    profit = 0.0
    for a, p_vals in solution["p"].items():
        for t, p in p_vals.items():
            profit += spot.iloc[t] * p * dt
    for a, pr_vals in solution["pr_up"].items():
        for t, pr in pr_vals.items():
            profit += reserve.iloc[t][0] * pr
    for a, pr_vals in solution["pr_dn"].items():
        for t, pr in pr_vals.items():
            profit += reserve.iloc[t][1] * pr
    for a, pb_vals in solution["pb_up"].items():
        for t, pb in pb_vals.items():
            profit += activation.iloc[t][0] * pb
    for a, pb_vals in solution["pb_dn"].items():
        for t, pb in pb_vals.items():
            profit += activation.iloc[t][1] * pb
    for a, s_vals in solution["s_up"].items():
        for t, s in s_vals.items():
            profit -= config.PENALTY * s
    for a, s_vals in solution["s_dn"].items():
        for t, s in s_vals.items():
            profit -= config.PENALTY * s
    return profit

def run_evaluation(min_lists, max_lists, spot, reserve, activation, indicators):
    """
    1) build offers
    2) schedule (joint/sequential, modules)
    3) compute profit
    4) return DataFrame with config + profit
    """
    offers = create_aggregated_offers(min_lists, max_lists)
    sol    = optimize(offers, spot, reserve, activation, indicators)
    profit = compute_profit(sol, spot, reserve, activation)
    row = {
        "mode": config.MODE,
        "run_spot":  config.RUN_SPOT,
        "run_reserve": config.RUN_RESERVE,
        "run_activation": config.RUN_ACTIVATION,
        "profit": profit
    }
    return pd.DataFrame([row])
