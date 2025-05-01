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

def optimize_flexoffers(offers, spot_prices, reserve_prices=None, activation_prices=None, indicators=None, fixed_p=None):
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
    #T = pad_profiles_to_common_timeline(offers)
    res = config.TIME_RESOLUTION 
    A = len(offers)

    earliest = min(fo.get_est() for fo in offers)
    offsets = [ int((fo.get_est()-earliest)/res) for fo in offers ]
    T = max(offsets[i] + len(offers[i].get_profile()) for i in range(A))

    prob = pulp.LpProblem("FlexOfferScheduling", pulp.LpMaximize)

    # Decision variables
        # p = {(a,t): pulp.LpVariable(f"p_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        # # --- if we're in sequential mode, lock in the spot‐schedule from the first solve:
        # if fixed_p is not None:
        #     for a in range(A):
        #         for t in range(T):
        #             prob += p[(a,t)] == fixed_p[a][t], f"fix_p_{a}_{t}"
    p = {}
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        for j, ts in enumerate(prof):
            t = offsets[a] + j
            p[(a,t)] = pulp.LpVariable(f"p_{a}_{t}", lowBound=0)

    if config.RUN_RESERVE:
    #     pr_up = {(a,t): pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    #     pr_dn = {(a,t): pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    # else:
    #     pr_up = pr_dn = {}
        pr_up = {}
        pr_dn = {}
        for a, fo in enumerate(offers):
            prof = fo.get_profile()
            for j in range(len(prof)):
                t = offsets[a] + j
                pr_up[(a,t)] = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
                pr_dn[(a,t)] = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)

    if config.RUN_ACTIVATION:
    #     pb_up = {(a,t): pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    #     pb_dn = {(a,t): pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    #     s_up  = {(a,t): pulp.LpVariable(f"s_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    #     s_dn  = {(a,t): pulp.LpVariable(f"s_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
    # else:
    #     pb_up = pb_dn = s_up = s_dn = {}
        pb_up = {}
        pb_dn = {}
        s_up  = {}
        s_dn  = {}
        for a, fo in enumerate(offers):
            prof = fo.get_profile()
            for j in range(len(prof)):
                t = offsets[a] + j
                pb_up[(a,t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
                pb_dn[(a,t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
                s_up[(a,t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
                s_dn[(a,t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    # Objective function
    dt = config.TIME_RESOLUTION / 3600.0
    obj = []

    for (a,t), var in p.items():
        spot = spot_prices.iloc[t]
        obj.append(-spot * var * dt)
        if config.RUN_RESERVE:
            r_up, r_dn = reserve_prices.iloc[t]
            obj.append(r_up * pr_up[(a,t)] * dt + r_dn * pr_dn[(a,t)] * dt)
        if config.RUN_ACTIVATION:
            b_up, b_dn = activation_prices.iloc[t]
            obj.append(b_up * pb_up[(a,t)] * dt + b_dn * pb_dn[(a,t)] * dt)
            obj.append(-config.PENALTY * (s_up[(a,t)] + s_dn[(a,t)]))
    prob += pulp.lpSum(obj)

    # Constraints
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        # 1) total‐energy only over valid slots
        prob += (pulp.lpSum(
                    p[(a, offsets[a]+j)] * dt
                    for j in range(len(prof))
                ) >= fo.get_min_overall_alloc()), f"total_min_energy_{a}"
        prob += (pulp.lpSum(
                    p[(a, offsets[a]+j)] * dt
                    for j in range(len(prof))
                ) <= fo.get_max_overall_alloc()), f"total_max_energy_{a}"

        # 2) slice‐by‐slice bounds
        for j, ts in enumerate(prof):
            t = offsets[a] + j
            prob += p[(a,t)] >= ts.min_power
            prob += p[(a,t)] <= ts.max_power

            if config.RUN_RESERVE:
                prob += pr_up[(a,t)] <= p[(a,t)]
                prob += pr_dn[(a,t)] <= ts.max_power - p[(a,t)]

            if config.RUN_ACTIVATION:
                d_up, d_dn = indicators.iloc[t]
                prob += pb_up[(a,t)] + s_up[(a,t)] >= pr_up[(a,t)] * d_up
                prob += pb_dn[(a,t)] + s_dn[(a,t)] >= pr_dn[(a,t)] * d_dn

    # Solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # Build output
    sol = {"p": {}, "pr_up": {}, "pr_dn": {}, "pb_up": {}, "pb_dn": {}, "s_up": {}, "s_dn": {}}
    
    for a in range(A):
        sol["p"][a] = {}
        if config.RUN_RESERVE:
            sol["pr_up"][a] = {}
            sol["pr_dn"][a] = {}
        if config.RUN_ACTIVATION:
            sol["pb_up"][a] = {}
            sol["pb_dn"][a] = {}
            sol["s_up"][a]  = {}
            sol["s_dn"][a]  = {}

    # Fill in the values for each variable you created
    for (a, t), var in p.items():
        sol["p"][a][t] = pulp.value(var)

    if config.RUN_RESERVE:
        for (a, t), var in pr_up.items():
            sol["pr_up"][a][t] = pulp.value(var)
        for (a, t), var in pr_dn.items():
            sol["pr_dn"][a][t] = pulp.value(var)

    if config.RUN_ACTIVATION:
        for (a, t), var in pb_up.items():
            sol["pb_up"][a][t] = pulp.value(var)
        for (a, t), var in pb_dn.items():
            sol["pb_dn"][a][t] = pulp.value(var)
        for (a, t), var in s_up.items():
            sol["s_up"][a][t]  = pulp.value(var)
        for (a, t), var in s_dn.items():
            sol["s_dn"][a][t]  = pulp.value(var)

    return sol



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
