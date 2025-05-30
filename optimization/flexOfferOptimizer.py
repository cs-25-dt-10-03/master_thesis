from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpMaximize, LpAffineExpression
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from database.dataManager import get_prices_in_range, fetch_mFRR_by_range, fetch_Regulating_by_range, load_and_prepare_prices
import pandas as pd
import csv
import os
import numpy as np

import pandas as pd
import pulp
from typing import List, Union
import pulp
from config import config
import time
from flexoffer_logic import Flexoffer, TimeSlice, DFO, padDFOsToCommonTimeline


def optimize_offers(offers, *args, **kwargs):
    """
    Dispatch to the right optimizer based on the type of the first offer.
    """
    if not offers:
        raise ValueError("No offers passed to optimize_offers()")
    first = offers[0]
    if isinstance(first, DFO):
        return optimize_dfos(offers, *args, **kwargs)
    elif isinstance(first, Flexoffer):
        return optimize_flexoffers(offers, *args, **kwargs)
    else:
        raise TypeError(f"Unknown offer type: {type(first)}")


def optimize_flexoffers(offers, spot_prices, reserve_prices=None, activation_prices=None, indicators=None, fixed_p=None, fixed_pr_up=None, fixed_pr_dn=None):
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
    # 1) Offsets & horizon
    
    config_start_ts = int(pd.to_datetime(config.SIMULATION_START_DATE).timestamp())
    min_offer_ts   = min(fo.get_est() for fo in offers)
    sim_start_ts   = min(config_start_ts, min_offer_ts)

    T = len(spot_prices)           # only slots we actually have prices for
    offsets = [int((fo.get_est() - sim_start_ts) / res) for fo in offers]

    # ensure every offer’s allowed window + duration fits in [0, T)
    for a, fo in enumerate(offers):
        dur     = fo.get_duration()
        est_ts  = fo.get_est()
        lst_ts  = fo.get_lst()
        est_idx = int((est_ts - sim_start_ts) / res)
        lst_idx = int((lst_ts - sim_start_ts) / res)
        if est_idx < 0 or lst_idx + dur > T:
            raise ValueError(
                f"FlexOffer {a} window [{est_idx}, {lst_idx + dur}) outside price horizon of length {T}")


    prob = pulp.LpProblem("FlexOfferScheduling", pulp.LpMaximize)

    #only one start time constraint
    z = {}  # z[a,s]: binary start time selector
    allowed_starts = {}  # Map of allowed start times for each offer
    for a, fo in enumerate(offers):
        allowed = [s for s in fo.get_allowed_start_times() if s + fo.get_duration() * res <= fo.get_et()]
        allowed_starts[a] = allowed
        for s_idx, s in enumerate(allowed):
            z[(a, s_idx)] = pulp.LpVariable(f"z_{a}_{s_idx}", cat='Binary')


    p = {}
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        allowed = allowed_starts[a]
        for s_idx, s in enumerate(allowed):
            for j, ts in enumerate(prof):
                t = int((s - sim_start_ts) / res) + j
                if t < T:
                    # Create power variable
                    var = pulp.LpVariable(f"p_{a}_{s_idx}_{t}", lowBound=0, upBound=ts.max_power)
                    p[(a, s_idx, t)] = var

                    # Add conditional bounds linked to binary start selection
                    prob += var <= ts.max_power * z[(a, s_idx)], f"max_bound_{a}_{s_idx}_{j}"
                    prob += var >= ts.min_power * z[(a, s_idx)], f"min_bound_{a}_{s_idx}_{j}"

    if fixed_p is not None:
        for (a, s_idx, t), var in p.items():
            val = fixed_p[a].get(t, 0.0)
            var.lowBound = val
            var.upBound  = val

    if config.RUN_RESERVE:
        pr_up = {}
        pr_dn = {}
        for a, fo in enumerate(offers):
            prof = fo.get_profile()
            for j in range(len(prof)):
                t = offsets[a] + j
                if t < T:
                    # create reserve vars
                    var_up = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
                    var_dn = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)

                    # if we have fixed reserve allocations from a prior pass, pin them
                    if fixed_pr_up is not None:
                        val_up = fixed_pr_up[a].get(t, 0.0)
                        var_up.lowBound = var_up.upBound = val_up
                    if fixed_pr_dn is not None:
                        val_dn = fixed_pr_dn[a].get(t, 0.0)
                        var_dn.lowBound = var_dn.upBound = val_dn

                    pr_up[(a,t)] = var_up
                    pr_dn[(a,t)] = var_dn
    if config.RUN_ACTIVATION:
        pb_up = {}
        pb_dn = {}
        s_up  = {}
        s_dn  = {}
        for a, fo in enumerate(offers):
            prof = fo.get_profile()
            for j in range(len(prof)):
                t = offsets[a] + j
                if t < T:
                    pb_up[(a,t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
                    pb_dn[(a,t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
                    s_up[(a,t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
                    s_dn[(a,t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    # Objective function
    dt = config.TIME_RESOLUTION / 3600.0
    obj = []

    for (a, s_idx, t), var in p.items():
        spot = spot_prices[t]
        #print(spot)
        obj.append(-spot * var * dt)
        if config.RUN_RESERVE and reserve_prices is not None:
            if (a, t) in pr_up and (a, t) in pr_dn: 
                r_up, r_dn, _, _ = reserve_prices.iloc[t]
                obj.append(r_up * pr_up[(a,t)] * dt + r_dn * pr_dn[(a,t)] * dt)
                if config.RUN_ACTIVATION and activation_prices is not None:
                    b_up, b_dn, penalty, _, _ = activation_prices.iloc[t]
                    d_up, d_dn = indicators[t]

                    # Only apply activation price if actually activated
                    b_up = b_up if d_up == 1 else 0
                    b_dn = b_dn if d_dn == 1 else 0

                    # Incremental activation revenue only
                    obj.append((b_up - spot) * pb_up[(a,t)] * dt)
                    obj.append((b_dn - spot) * pb_dn[(a,t)] * dt)
                    obj.append(-penalty * (s_up[(a,t)] + s_dn[(a,t)]) * dt)
    prob += pulp.lpSum(obj)

    # Constraints
    # --- only one start time ---
    for a in range(len(offers)):
        prob += pulp.lpSum(z[(a, s_idx)] for s_idx in range(len(allowed_starts[a]))) == 1

    # --- min total energy ---
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        min_energy = []
        max_energy = []
        for s_idx, s in enumerate(allowed_starts[a]):
            for j in range(len(prof)):
                t = int((s - sim_start_ts) / res) + j
                if t < T and (a, s_idx, t) in p:
                    min_energy.append(p[(a, s_idx, t)] * dt)
                    max_energy.append(p[(a, s_idx, t)] * dt)
        prob += pulp.lpSum(min_energy) >= fo.get_min_overall_alloc(), f"total_min_energy_{a}"
        prob += pulp.lpSum(max_energy) <= fo.get_max_overall_alloc(), f"total_max_energy_{a}"

        # 2) slice‐by‐slice bounds
        for j, ts in enumerate(prof):
            t = offsets[a] + j
            if config.RUN_RESERVE:
                if t < T:
                    p_total = pulp.lpSum(p[(a, s_idx, t)] for s_idx in range(len(allowed_starts[a])) if (a, s_idx, t) in p)
                    prob += pr_dn[(a,t)] <= ts.max_power - p_total
                    prob += pr_up[(a,t)] <= ts.max_power

            if config.RUN_ACTIVATION and indicators is not None:
                if t < T:

                    prob += pb_up[(a,t)] <= pr_up[(a,t)]
                    prob += pb_dn[(a,t)] <= pr_dn[(a,t)]

                    d_up, d_dn = indicators[t]
                    if d_up == 0:
                        prob += pb_up[(a,t)] == 0
                    if d_dn == 0:
                        prob += pb_dn[(a,t)] == 0

                    prob += pb_up[(a,t)] + s_up[(a,t)] >= pr_up[(a,t)] * d_up
                    prob += pb_dn[(a,t)] + s_dn[(a,t)] >= pr_dn[(a,t)] * d_dn

                    if j > 0:
                        t_prev = offsets[a] + j - 1
                        if (a, t) in pr_up and (a, t_prev) in pr_up:
                            dt_minutes = config.TIME_RESOLUTION / 60

                            ramp_limit_up = ts.max_power * (10 / dt_minutes)
                            ramp_limit_dn = ts.max_power * (10 / dt_minutes)
                            prob += pr_up[(a, t)] - pr_up[(a, t_prev)] <= ramp_limit_up
                            prob += pr_dn[(a, t)] - pr_dn[(a, t_prev)] <= ramp_limit_dn
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
    for a in range(A):
        sol["p"][a] = {}
        for s_idx, s in enumerate(allowed_starts[a]):
            for j in range(len(offers[a].get_profile())):
                t = int((s - sim_start_ts) / res) + j
                key = (a, s_idx, t)
                if key in p:
                    val = pulp.value(p[key])
                    if val > 0:
                        sol["p"][a][t] = val

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

    for a, fo in enumerate(offers):
        # Find chosen start time
        chosen_s_idx = None
        allowed = allowed_starts[a]
        for s_idx in range(len(allowed)):
            if pulp.value(z[(a, s_idx)]) > 0.5:
                chosen_s_idx = s_idx
                break

        if chosen_s_idx is not None:
            start_time = allowed[chosen_s_idx]
            fo.set_scheduled_start_time(start_time)

            allocation = [0.0] * fo.get_duration()
            for j in range(fo.get_duration()):
                t = int((start_time - sim_start_ts) / res) + j
                key = (a, chosen_s_idx, t)
                if key in p:
                    val = pulp.value(p[key])
                    allocation[j] = val
            fo.set_scheduled_allocation(allocation)

    return sol






















def optimize_dfos(
    dfos: List[DFO],
    spot_prices: Union[pd.Series, list],
    reserve_prices: Union[pd.DataFrame, list]=None,
    activation_prices: Union[pd.DataFrame, list]=None,
    indicators: Union[pd.DataFrame, list]=None,
    fixed_p: dict=None, 
    fixed_pr_up: dict=None, 
    fixed_pr_dn: dict=None
):

    # —————————————————————————————————————————————————————————————
    # 1) Offsets & horizon
    # —————————————————————————————————————————————————————————————
    res      = config.TIME_RESOLUTION
    sim_start_ts = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    T = len(spot_prices)           # only slots we actually have prices for

    offsets = [
        int((fo.get_est() - sim_start_ts) / res)
        for fo in dfos
    ]

    A        = len(dfos)
    dt       = res/3600.0

    # —————————————————————————————————————————————————————————————
    # 2) Decision vars only for valid (a,t)
    # —————————————————————————————————————————————————————————————
    p      = {}
    pr_up  = {}
    pr_dn  = {}
    pb_up  = {}
    pb_dn  = {}
    s_up   = {}
    s_dn   = {}

    for a, d in enumerate(dfos):
        # cumulative‐dependency var
        for j, poly in enumerate(d.polygons):
            t = offsets[a] + j
            if t < T:
                # slot‐energy
                p[(a,t)] = pulp.LpVariable(f"p_{a}_{t}", lowBound=0)
                # reserve
                if config.RUN_RESERVE:
                    # create reserve vars
                    var_up = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
                    var_dn = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)

                    # if we have fixed reserve allocations from a prior pass, pin them
                    if fixed_pr_up is not None:
                        val_up = fixed_pr_up[a].get(t, 0.0)
                        var_up.lowBound = var_up.upBound = val_up
                    if fixed_pr_dn is not None:
                        val_dn = fixed_pr_dn[a].get(t, 0.0)
                        var_dn.lowBound = var_dn.upBound = val_dn

                    pr_up[(a,t)] = var_up
                    pr_dn[(a,t)] = var_dn
                # activation & penalty
                if config.RUN_ACTIVATION:
                    pb_up[(a,t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
                    pb_dn[(a,t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
                    s_up[(a,t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
                    s_dn[(a,t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    # —————————————————————————————————————————————————————————————
    # 3) Build objective (spot, reserve, activation)
    # —————————————————————————————————————————————————————————————
    prob = pulp.LpProblem("DFO_scheduling", pulp.LpMaximize)
    terms = []
    for (a,t), var in p.items():
        price_spot = spot_prices.iloc[t]
        terms.append(- price_spot * var * dt)

        if config.RUN_RESERVE and reserve_prices is not None:
            ru, rd = reserve_prices.iloc[t]
            terms.append((ru * pr_up[(a,t)] + rd * pr_dn[(a,t)]) * dt)

        if config.RUN_ACTIVATION and activation_prices is not None:
            bu, bd = activation_prices.iloc[t]
            terms.append((bu * pb_up[(a,t)] + bd * pb_dn[(a,t)]) * dt)
            terms.append(- config.PENALTY * (s_up[(a,t)] + s_dn[(a,t)]) * dt)

    prob += pulp.lpSum(terms)

    # —————————————————————————————————————————————————————————————
    # 5) Dependency‐slice & coupling constraints
    # —————————————————————————————————————————————————————————————
    for a, d in enumerate(dfos):
        cumulative_energy = LpAffineExpression()
        poly_list = d.polygons  # each is a DependencyPolygon

        min_terms = []
        max_terms = []
        for j in range(len(d.polygons)):
            t = offsets[a] + j
            if t < T:
                min_terms.append(p[(a, t)] * dt)
                max_terms.append(p[(a, t)] * dt)
        if min_terms:
            prob += (pulp.lpSum(min_terms)
                     >= d.min_total_energy), f"min_total_{a}"
        if max_terms:
            prob += (pulp.lpSum(max_terms)
                     <= d.max_total_energy), f"max_total_{a}"
        
        # print(f"min og max energy: {d.min_total_energy} og {d.max_total_energy}")



        for j, poly in enumerate(poly_list):
            t = offsets[a] + j
            if t >= T:
                continue
        
            points = poly.points 

            if len(points) < 4:
                energy_min = points[0].y
                energy_max = points[1].y
            else:
                for k in range(1, len(points) - 2, 2):
                    prev_min = points[k - 1]
                    prev_max = points[k]
                    next_min = points[k + 1]
                    next_max = points[k + 2]

                    if next_min.x == prev_min.x or next_max.x == prev_max.x:
                        continue

                    energy_min = prev_min.y + ((next_min.y - prev_min.y) / (next_min.x - prev_min.x)) * (cumulative_energy - prev_min.x)
                    energy_max = prev_max.y + ((next_max.y - prev_max.y) / (next_max.x - prev_max.x)) * (cumulative_energy - prev_max.x)
                    break

            if config.RUN_SPOT:
                prob += p[(a, t)] >= energy_min, f"min_energy_{a}_{t}"
                prob += p[(a, t)] <= energy_max, f"max_energy_{a}_{t}"

            if config.RUN_RESERVE:
                if config.RUN_SPOT:
                    prob += pr_up[(a, t)] <= p[(a, t)], f"r_up_limit_{a}_{t}"
                    prob += pr_dn[(a, t)] <= energy_max - p[(a, t)], f"r_dn_limit_{a}_{t}"
                else:
                    prob += pr_up[(a, t)] <= energy_max, f"r_up_max_{a}_{t}"
                    prob += pr_dn[(a, t)] <= energy_max, f"r_dn_max_{a}_{t}"

            if config.RUN_ACTIVATION and indicators is not None:
                delta_up, delta_dn = indicators[t]
                prob += pb_up[(a, t)] + s_up[(a, t)] >= pr_up[(a, t)] * delta_up
                prob += pb_dn[(a, t)] + s_dn[(a, t)] >= pr_dn[(a, t)] * delta_dn

            if config.RUN_SPOT:
                cumulative_energy += p[(a, t)]

    # —————————————————————————————————————————————————————————————
    # 6) Fix spot‐only in sequential mode
    # —————————————————————————————————————————————————————————————
    if fixed_p is not None:
        for (a,t), var in p.items():
            val = fixed_p[a].get(t, 0.0)
            var.lowBound = val
            var.upBound  = val

    # —————————————————————————————————————————————————————————————
    # 7) Solve & extract exactly as your flexoffer version
    # —————————————————————————————————————————————————————————————
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    sol = {k:{a:{} for a in range(A)} for k in ("p","pr_up","pr_dn","pb_up","pb_dn","s_up","s_dn")}
    for (a,t), var in p.items(): sol["p"][a][t] = pulp.value(var)
    if config.RUN_RESERVE:
        for (a,t), var in pr_up.items(): sol["pr_up"][a][t] = pulp.value(var)
        for (a,t), var in pr_dn.items(): sol["pr_dn"][a][t] = pulp.value(var)
    if config.RUN_ACTIVATION:
        for (a,t), var in pb_up.items(): sol["pb_up"][a][t] = pulp.value(var)
        for (a,t), var in pb_dn.items(): sol["pb_dn"][a][t] = pulp.value(var)
        for (a,t), var in s_up.items(): sol["s_up"][a][t] = pulp.value(var)
        for (a,t), var in s_dn.items(): sol["s_dn"][a][t] = pulp.value(var)

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
