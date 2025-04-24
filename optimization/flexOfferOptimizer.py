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


def optimize(offers: List[Flexoffer]) -> List[Flexoffer]:
    """
    Dynamically build LP alt efter config indstillinger.
    offers: list of AFOs
    spot_prices: Dataframe: [HourDK, SpotPriceDKK]
    reserve_prices: Dataframe: [HourDK, mFRR_UpPriceDKK, mFRR_DownPriceDKK]
    activation_prices: Dataframe: [HourDK, UpBalancingPriceDKK, DownBalancingPriceDKK]
    """

    earliest_start = min(o.get_est() for o in offers)

    A = len(offers)
    T = max(len(o.get_profile()) for o in offers)

    spot_prices, reserve_prices, activation_prices, indicators = load_and_prepare_prices(earliest_start, T, resolution=config.TIME_RESOLUTION)

    # PAD profiles så alle har længde T
    for o in offers:
        prof = o.get_profile()
        prof.extend([TimeSlice(0,0)] * (T - len(prof)))

    def build_and_solve(use_spot, use_reserve, use_activation, fixed_p=None):
        prob = pulp.LpProblem("scheduling", pulp.LpMaximize)

        # --- Variables ---
        if use_spot:
            p = {(a,t): pulp.LpVariable(f"p_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        else:
            p = None

        if use_reserve:
            pr_up = {(a,t): pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
            pr_dn = {(a,t): pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        else:
            pr_up = pr_dn = None

        if use_activation:
            pb_up = {(a,t): pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
            pb_dn = {(a,t): pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
            s_up  = {(a,t): pulp.LpVariable(f"s_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
            s_dn  = {(a,t): pulp.LpVariable(f"s_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)}
        else:
            pb_up = pb_dn = s_up = s_dn = None

        # --- Objective ---
        obj = []
        for t in range(T):
            if use_spot:
                spot = spot_prices.iloc[t]
            if use_reserve:
                r_up, r_dn = reserve_prices.iloc[t]
            if use_activation:
                b_up, b_dn = activation_prices.iloc[t]

            for a in range(A):
                if use_reserve:
                    obj.append(r_up * pr_up[(a,t)] + r_dn * pr_dn[(a,t)])
                if use_activation:
                    obj.append(b_up * pb_up[(a,t)] + b_dn * pb_dn[(a,t)])
                if use_spot:
                    dt = config.TIME_RESOLUTION / 3600.0
                    obj.append(-spot * p[(a,t)] * dt)
                if use_activation:
                    obj.append(- config.PENALTY * (s_up[(a,t)] + s_dn[(a,t)]))

        prob += pulp.lpSum(obj)

        # --- Constraints ---
        for a in range(A):
            prof = offers[a].get_profile()
            for t in range(T):
                pmin, pmax = prof[t].min_power, prof[t].max_power
                if use_spot:
                    prob += p[(a,t)] >= pmin
                    prob += p[(a,t)] <= pmax
                if use_reserve:
                    if use_spot:
                        prob += pr_up[(a,t)] <= p[(a,t)]
                        prob += pr_dn[(a,t)] <= pmax - p[(a,t)]
                    else:
                        # hvis ingen spot, brug min_power=0
                        prob += pr_up[(a,t)] <= pmax
                        prob += pr_dn[(a,t)] <= pmax
                if use_activation:
                    delta_up, delta_dn = indicators[t]
                    prob += pb_up[(a,t)] + s_up[(a,t)] >= pr_up[(a,t)] * delta_up
                    prob += pb_dn[(a,t)] + s_dn[(a,t)] >= pr_dn[(a,t)] * delta_dn

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # --- Extract ---
        sol = {"p":{}, "pr_up":{}, "pr_dn":{}, "pb_up":{}, "pb_dn":{}, "s_up":{}, "s_dn":{}}
        for a in range(A):
            if use_spot:
                sol["p"][a] = { t: pulp.value(p[(a,t)]) for t in range(T) }
            if use_reserve:
                sol["pr_up"][a] = { t: pulp.value(pr_up[(a,t)]) for t in range(T) }
                sol["pr_dn"][a] = { t: pulp.value(pr_dn[(a,t)]) for t in range(T) }
            if use_activation:
                sol["pb_up"][a] = { t: pulp.value(pb_up[(a,t)]) for t in range(T) }
                sol["pb_dn"][a] = { t: pulp.value(pb_dn[(a,t)]) for t in range(T) }
                sol["s_up"][a]  = { t: pulp.value(s_up[(a,t)])  for t in range(T) }
                sol["s_dn"][a]  = { t: pulp.value(s_dn[(a,t)])  for t in range(T) }
        return sol



    # --- Joint eller Sequential --- #
    if config.MODE == "joint":
        return build_and_solve(
            use_spot       = config.RUN_SPOT,
            use_reserve    = config.RUN_RESERVE,
            use_activation = config.RUN_ACTIVATION)
    
    elif config.MODE == "sequential":
        # Første LP: kun spot/reserve aktiveret iht. config
        sol1 = build_and_solve(
            use_spot       = config.RUN_SPOT,
            use_reserve    = config.RUN_RESERVE,
            use_activation = False
        )
        # Gem p1
        p1 = sol1["p"]
        # Anden LP: kun reserve+activation (spot fast)
        sol2 = build_and_solve(
            use_spot       = False,
            use_reserve    = config.RUN_RESERVE,
            use_activation = config.RUN_ACTIVATION,
            fixed_p        = p1
        )
        
        # Merge: p fra sol1, alt andet fra sol2
        merged = sol1.copy()
        for key in ["pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn"]:
            merged[key] = sol2[key]
        return merged
    else:
        raise ValueError("Unknown MODE")