from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from database.dataManager import get_prices_in_range, fetch_mFRR_by_range, fetch_Regulating_by_range, load_and_prepare_prices
import pandas as pd
import csv
import os
import numpy as np
import pulp
from config import config
import time
from flexoffer_logic import Flexoffer, TimeSlice


# def optimize(offers: List[Flexoffer]) -> List[Flexoffer]:
#     """
#     Dynamically build LP alt efter config indstillinger.
#     offers: list of AFOs
#     spot_prices: Dataframe: [HourDK, SpotPriceDKK]
#     reserve_prices: Dataframe: [HourDK, mFRR_UpPriceDKK, mFRR_DownPriceDKK]
#     activation_prices: Dataframe: [HourDK, UpBalancingPriceDKK, DownBalancingPriceDKK]
#     """

#     earliest_start = min(fo.get_est() for fo in offers)
#     T = pad_profiles_to_common_timeline(offers)
#     A = len(offers)
#     dt = config.TIME_RESOLUTION / 3600.0

#     spot_prices, reserve_prices, activation_prices, indicators = load_and_prepare_prices(earliest_start, T, resolution=config.TIME_RESOLUTION)

#     def build_and_solve(use_spot, use_reserve, use_activation, fixed_p=None):
#         prob = pulp.LpProblem("scheduling", pulp.LpMaximize)

#         # --- Variables ---
#         if use_spot:
#             p = {(a,t): pulp.LpVariable(f"p_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#         else:
#             p = None

#         if use_reserve:
#             pr_up = {(a,t): pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#             pr_dn = {(a,t): pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#         else:
#             pr_up = pr_dn = None

#         if use_activation:
#             pb_up = {(a,t): pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#             pb_dn = {(a,t): pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#             s_up  = {(a,t): pulp.LpVariable(f"s_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#             s_dn  = {(a,t): pulp.LpVariable(f"s_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
#         else:
#             pb_up = pb_dn = s_up = s_dn = None

#         # --- Objective ---
#         obj = []
#         for t in range(T):
#             if use_spot:
#                 spot = spot_prices.iloc[t]
#             if use_reserve:
#                 r_up, r_dn = reserve_prices.iloc[t]
#             if use_activation:
#                 b_up, b_dn = activation_prices.iloc[t]

#             for a in range(A):
#                 if use_reserve:
#                     obj.append(r_up * pr_up[(a,t)] + r_dn * pr_dn[(a,t)])
#                 if use_activation:
#                     obj.append(b_up * pb_up[(a,t)] + b_dn * pb_dn[(a,t)])
#                 if use_spot:
#                     obj.append(-spot * p[(a,t)] * dt)
#                 if use_activation:
#                     obj.append(- config.PENALTY * (s_up[(a,t)] + s_dn[(a,t)]))

#         prob += pulp.lpSum(obj)

#         # --- Constraints ---
#         for a in range(A):
#             prof = offers[a].get_profile()
#             for t in range(T):
#                 pmin, pmax = prof[t].min_power, prof[t].max_power
#                 if use_spot:
#                     prob += p[(a,t)] >= pmin
#                     prob += p[(a,t)] <= pmax
#                 if use_reserve:
#                     if use_spot:
#                         prob += pr_up[(a,t)] <= p[(a,t)]
#                         prob += pr_dn[(a,t)] <= pmax - p[(a,t)]
#                     else:
#                         # hvis ingen spot, brug min_power=0
#                         prob += pr_up[(a,t)] <= pmax
#                         prob += pr_dn[(a,t)] <= pmax
#                 if use_activation:
#                     delta_up, delta_dn = indicators[t]
#                     prob += pb_up[(a,t)] + s_up[(a,t)] >= pr_up[(a,t)] * delta_up
#                     prob += pb_dn[(a,t)] + s_dn[(a,t)] >= pr_dn[(a,t)] * delta_dn

#         prob.solve(pulp.PULP_CBC_CMD(msg=False))

#         # --- Extract ---
#         sol = {"p":{}, "pr_up":{}, "pr_dn":{}, "pb_up":{}, "pb_dn":{}, "s_up":{}, "s_dn":{}}
#         for a in range(A):
#             if use_spot:
#                 sol["p"][a] = { t: pulp.value(p[(a,t)]) for t in range(T) }
#             if use_reserve:
#                 sol["pr_up"][a] = { t: pulp.value(pr_up[(a,t)]) for t in range(T) }
#                 sol["pr_dn"][a] = { t: pulp.value(pr_dn[(a,t)]) for t in range(T) }
#             if use_activation:
#                 sol["pb_up"][a] = { t: pulp.value(pb_up[(a,t)]) for t in range(T) }
#                 sol["pb_dn"][a] = { t: pulp.value(pb_dn[(a,t)]) for t in range(T) }
#                 sol["s_up"][a]  = { t: pulp.value(s_up[(a,t)])  for t in range(T) }
#                 sol["s_dn"][a]  = { t: pulp.value(s_dn[(a,t)])  for t in range(T) }



#         for a, fo in enumerate(offers):
#             alloc = [sol["p"][a][t] for t in range(T)]
#             fo.set_scheduled_allocation(alloc)

#             # find første t med non-zero for start time
#             try:
#                 first = next(t for t, v in enumerate(alloc) if v > 1e-6)
#             except StopIteration:
#                 first = 0
#             fo.set_scheduled_start_time(earliest_start + first * config.TIME_RESOLUTION)

#         return sol



#     # --- Joint eller Sequential --- #
#     if config.MODE == "joint":
#         return build_and_solve(
#             use_spot       = config.RUN_SPOT,
#             use_reserve    = config.RUN_RESERVE,
#             use_activation = config.RUN_ACTIVATION)
    
#     elif config.MODE == "sequential":
#         # Første LP: kun spot/reserve aktiveret iht. config
#         sol1 = build_and_solve(
#             use_spot       = config.RUN_SPOT,
#             use_reserve    = config.RUN_RESERVE,
#             use_activation = False
#         )
#         # Gem p1
#         p1 = sol1["p"]
#         # Anden LP: kun reserve+activation (spot fast)
#         sol2 = build_and_solve(
#             use_spot       = False,
#             use_reserve    = config.RUN_RESERVE,
#             use_activation = config.RUN_ACTIVATION,
#             fixed_p        = p1
#         )
        
#         # Merge: p fra sol1, alt andet fra sol2
#         merged = sol1.copy()
#         for key in ["pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"]:
#             merged[key] = sol2[key]
#         return merged
#     else:
#         raise ValueError("Unknown MODE")



def pad_profiles_to_common_timeline(offers: List[Flexoffer]) -> int:
    """
    Pads all FlexOffers in-place so their profiles align on a common time axis.
    This ensures no index errors during optimization.

    offers: List of flexOffers
    Returns:
        T (int): common time length
    """
    res = config.TIME_RESOLUTION
    earliest = min(fo.get_est() for fo in offers)
    
    offsets = [
        int((fo.get_est() - earliest) / res)
        for fo in offers
    ]
    
    T = max(offset + len(fo.get_profile()) for offset, fo in zip(offsets, offers))

    for offset, fo in zip(offsets, offers):
        profile = fo.get_profile()
        pre = [TimeSlice(0, 0)] * offset
        post = [TimeSlice(0, 0)] * (T - offset - len(profile))
        profile[:0] = pre      # insert at front
        profile.extend(post)   # append at end
        fo.set_profile(pre + profile + post)        

    return T

def optimize_flexoffers(offers, spot_prices, reserve_prices=None, activation_prices=None, indicators=None):
    """
    Build and solve LP or MILP scheduling for a set of FlexOffers.

    Args:
        offers (list): List of Flexoffer objects
        spot_prices (list): Spot prices (length T)
        reserve_prices (list of tuples): (up, down) reserve prices
        activation_prices (list of tuples): (up, down) activation prices
        indicators (list of tuples): (delta_up, delta_dn) activation indicators

    Returns:
        dict: Solution dictionary { "p": {a: {t: value}}, "pr_up": {...}, etc. }
    """

    # --- Build problem ---
    T = pad_profiles_to_common_timeline(offers)
    A = len(offers)

    prob = pulp.LpProblem("FlexOfferScheduling", pulp.LpMaximize)

    # Decision variables
    p = {(a,t): pulp.LpVariable(f"p_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}

    if config.RUN_RESERVE:
        pr_up = {(a,t): pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        pr_dn = {(a,t): pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    else:
        pr_up = pr_dn = {}

    if config.RUN_ACTIVATION:
        pb_up = {(a,t): pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        pb_dn = {(a,t): pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        s_up  = {(a,t): pulp.LpVariable(f"s_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        s_dn  = {(a,t): pulp.LpVariable(f"s_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    else:
        pb_up = pb_dn = s_up = s_dn = {}

    # Objective function
    dt = config.TIME_RESOLUTION / 3600.0
    obj = []

    for t in range(T):
        for a in range(A):
            spot = spot_prices.iloc[t]
            obj.append(-spot * p[(a,t)] * dt)
            if config.RUN_RESERVE:
                r_up, r_dn = reserve_prices.iloc[t]
                obj.append(r_up * pr_up[(a,t)] + r_dn * pr_dn[(a,t)])
            if config.RUN_ACTIVATION:
                b_up, b_dn = activation_prices.iloc[t]
                obj.append(b_up * pb_up[(a,t)] + b_dn * pb_dn[(a,t)])
                obj.append(-config.PENALTY * (s_up[(a,t)] + s_dn[(a,t)]))

    prob += pulp.lpSum(obj)

    # Constraints
    for a in range(A):
        profile = offers[a].get_profile()
        # --- total energy constraint ---
        min_energy = offers[a].get_min_overall_alloc()
        max_energy = offers[a].get_max_overall_alloc()

        prob += pulp.lpSum(p[(a, t)] * dt for t in range(T)) >= min_energy, f"total_min_energy_{a}"
        prob += pulp.lpSum(p[(a, t)] * dt for t in range(T)) <= max_energy, f"total_max_energy_{a}"

        for t in range(T):
            if t < len(profile):
                ts = profile[t]
                pmin, pmax = ts.min_power, ts.max_power
                prob += p[(a,t)] >= pmin
                prob += p[(a,t)] <= pmax

                if config.RUN_RESERVE:
                    prob += pr_up[(a,t)] <= p[(a,t)]
                    prob += pr_dn[(a,t)] <= pmax - p[(a,t)]

                if config.RUN_ACTIVATION:
                    d_up, d_dn = indicators[t]
                    prob += pb_up[(a,t)] + s_up[(a,t)] >= pr_up[(a,t)] * d_up
                    prob += pb_dn[(a,t)] + s_dn[(a,t)] >= pr_dn[(a,t)] * d_dn
            else:
                # If no valid profile at t, force p = 0
                prob += p[(a,t)] == 0

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Build output
    sol = {"p": {}, "pr_up": {}, "pr_dn": {}, "pb_up": {}, "pb_dn": {}, "s_up": {}, "s_dn": {}}
    for a in range(A):
        sol["p"][a] = {t: pulp.value(p[(a,t)]) for t in range(T)}
        if config.RUN_RESERVE:
            sol["pr_up"][a] = {t: pulp.value(pr_up[(a,t)]) for t in range(T)}
            sol["pr_dn"][a] = {t: pulp.value(pr_dn[(a,t)]) for t in range(T)}
        if config.RUN_ACTIVATION:
            sol["pb_up"][a] = {t: pulp.value(pb_up[(a,t)]) for t in range(T)}
            sol["pb_dn"][a] = {t: pulp.value(pb_dn[(a,t)]) for t in range(T)}
            sol["s_up"][a]  = {t: pulp.value(s_up[(a,t)]) for t in range(T)}
            sol["s_dn"][a]  = {t: pulp.value(s_dn[(a,t)]) for t in range(T)}

    return sol
