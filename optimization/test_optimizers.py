from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD
from datetime import datetime
import pandas as pd
import numpy as np
import os

from config import config
from flexoffer_logic import Flexoffer
from database.dataManager import loadSpotPriceData
import pulp

def sample_price_scenarios(horizon_len, num_scenarios):
    """
    Block-bootstrap S scenarios of length `horizon_len` from full historical series.
    Returns:
      scen_spots:   list of pandas.Series length horizon_len
      scen_reserve: list of DataFrame [(Up,Down)] length horizon_len
      scen_act:     list of DataFrame [(Up,Down)] length horizon_len
    """
    # 1) Full history
    full_spot = loadSpotPriceData()['SpotPriceDKK'].values
    # mFRR full file
    fname_mfrr = "mFRR_15min.csv" if config.TIME_RESOLUTION<3600 else "mFRR.csv"
    full_mfrr = pd.read_csv(os.path.join(config.DATA_FILEPATH, fname_mfrr),
                            usecols=['mFRR_UpPriceDKK','mFRR_DownPriceDKK']).values
    # activation full file
    fname_act = "Regulating_15min.csv" if config.TIME_RESOLUTION<3600 else "Regulating.csv"
    full_act = pd.read_csv(os.path.join(config.DATA_FILEPATH, fname_act),
                           usecols=['BalancingPowerPriceUpDKK','BalancingPowerPriceDownDKK']).values

    L = len(full_spot)
    scen_spots, scen_res, scen_act = [], [], []
    for _ in range(num_scenarios):
        # pick random start so that block of horizon_len fits
        start = np.random.randint(0, L - horizon_len + 1)
        scen_spots.append(pd.Series(full_spot[start:start+horizon_len]))
        scen_res.append(pd.DataFrame(full_mfrr[start:start+horizon_len],
                                     columns=['Up','Down']))
        scen_act.append(pd.DataFrame(full_act[start:start+horizon_len],
                                     columns=['Up','Down']))
    return scen_spots, scen_res, scen_act

def optimize_stochastic(
    offers, spot_prices, reserve_prices=None,
    activation_prices=None, indicators=None,
    num_scenarios=10
):
    """
    Two-stage stochastic:
     - Stage 1: choose day-ahead p[a,t]
     - Stage 2: recourse delta[a,t] settled at imbalance prices minus penalty
    Objective: minimize expected total cost across spot, reserve, and activation.
    """
    # Prepare
    S = num_scenarios
    T = len(spot_prices)
    dt = config.TIME_RESOLUTION / 3600.0
    sim0 = datetime.timestamp(pd.to_datetime(config.SIMULATION_START_DATE))
    offsets = [
        int((fo.get_est() - sim0)/config.TIME_RESOLUTION)
        for fo in offers
    ]

    # Sample scenarios
    scen_spots, scen_res, scen_act = sample_price_scenarios(T, S)

    # Build MILP
    prob = LpProblem("Stochastic_Scheduling", LpMaximize)
    # Stage-1 variables
    p = {}
    r_up = {}    # reserve capacity up
    r_dn = {}    # reserve capacity down
    a_up = {}    # activation up
    a_dn = {}    # activation down

    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        for j, ts in enumerate(prof):
            t = offsets[a] + j
            if t< T:
                p[(a,t)]    = LpVariable(f"p_{a}_{t}",    lowBound=ts.min_power, upBound=ts.max_power)
                r_up[(a,t)] = LpVariable(f"r_up_{a}_{t}", lowBound=0, upBound=ts.max_power)
                r_dn[(a,t)] = LpVariable(f"r_dn_{a}_{t}", lowBound=0, upBound=ts.max_power)
                a_up[(a,t)] = LpVariable(f"a_up_{a}_{t}", lowBound=0, upBound=ts.max_power)
                a_dn[(a,t)] = LpVariable(f"a_dn_{a}_{t}", lowBound=0, upBound=ts.max_power)

    # Stage-2 recourse: deviations on energy and reserve calls
    δ_p   = {}
    δ_ru  = {}
    δ_rd  = {}
    δ_au  = {}
    δ_ad  = {}

    for s in range(S):
        for (a,t), var in p.items():
            δ_p[(s,a,t)]  = LpVariable(f"dp_{s}_{a}_{t}", lowBound=-var.upBound, upBound=var.upBound)
            δ_ru[(s,a,t)] = LpVariable(f"dru_{s}_{a}_{t}", lowBound=-r_up[(a,t)].upBound, upBound=r_up[(a,t)].upBound)
            δ_rd[(s,a,t)] = LpVariable(f"drd_{s}_{a}_{t}", lowBound=-r_dn[(a,t)].upBound, upBound=r_dn[(a,t)].upBound)
            δ_au[(s,a,t)] = LpVariable(f"dau_{s}_{a}_{t}", lowBound=-a_up[(a,t)].upBound, upBound=a_up[(a,t)].upBound)
            δ_ad[(s,a,t)] = LpVariable(f"dad_{s}_{a}_{t}", lowBound=-a_dn[(a,t)].upBound, upBound=a_dn[(a,t)].upBound)

        # preserve totals (no net energy creation)
        for a, fo in enumerate(offers):
            prof = fo.get_profile()
            prob += lpSum(δ_p[(s,a,offsets[a]+j)] for j in range(len(prof))) == 0
            prob += lpSum(δ_ru[(s,a,offsets[a]+j)] for j in range(len(prof))) == 0
            prob += lpSum(δ_rd[(s,a,offsets[a]+j)] for j in range(len(prof))) == 0

    # Energy constraints stage-1
    for a, fo in enumerate(offers):
        prof = fo.get_profile()
        prob += lpSum(p[(a,offsets[a]+j)]*dt for j in range(len(prof))) >= fo.get_min_overall_alloc()
        prob += lpSum(p[(a,offsets[a]+j)]*dt for j in range(len(prof))) <= fo.get_max_overall_alloc()

    # Reserve capacity constraints
    # r_up <= p, r_dn <= max_power - p   (can only reserve what you’re not charging)
    for (a,t), var in p.items():
        prof = offers[a].get_profile()
        max_pow = prof[t-offsets[a]].max_power
        prob += r_up[(a,t)] <= var
        prob += r_dn[(a,t)] <= max_pow - var

    # Activation bounds: a_up <= r_up, a_dn <= r_dn
    for (a,t), ru_var in r_up.items():
        prob += a_up[(a,t)] <= ru_var
        prob += a_dn[(a,t)] <= r_dn[(a,t)]

    # Objective: **minimize** expected total cost
    # → maximize negative cost:
    obj = []

    # 1) Day‐ahead cost: spot + reserve capacity fees
    for (a,t), var in p.items():
        obj.append(-spot_prices.iloc[t] * var * dt)
        obj.append(-reserve_prices.iloc[t]['mFRR_UpPriceDKK']  * r_up[(a,t)] * dt)
        obj.append(-reserve_prices.iloc[t]['mFRR_DownPriceDKK']* r_dn[(a,t)] * dt)

    # 2) Expected recourse cost: activation fees + penalty on any mismatch
    penalty = config.PENALTY
    for s in range(S):
        sp = scen_spots[s]
        rr = scen_res[s]
        ac = scen_act[s]
        for (a,t), var in p.items():
            # recourse on energy: settled at activation price
            obj.append(-ac.iloc[t]['Up']  * (var + δ_p[(s,a,t)]) * dt)
            obj.append(-ac.iloc[t]['Down']* (var + δ_p[(s,a,t)]) * dt)
            # recourse on reserve calls: no separate recourse in mFRR
            # penalty for any shift
            obj.append(-penalty * abs(δ_p[(s,a,t)]) * dt)

    prob += lpSum(obj)

    # Solve
    prob.solve(PULP_CBC_CMD(msg=False))

    # Build solution dict
    sol = {'p':{}, 'pr_up':{}, 'pr_dn':{}, 'pb_up':{}, 'pb_dn':{}, 's_up':{}, 's_dn':{}}
    for a,_ in enumerate(offers):
        sol['p'][a], sol['pr_up'][a], sol['pr_dn'][a] = {},{},{}
        sol['pb_up'][a], sol['pb_dn'][a], sol['s_up'][a], sol['s_dn'][a] = {},{},{},{}
    for (a,t), var in p.items():
        sol['p'][a][t] = var.value()
        sol['pr_up'][a][t] = r_up[(a,t)].value()
        sol['pr_dn'][a][t] = r_dn[(a,t)].value()
        sol['pb_up'][a][t] = a_up[(a,t)].value()
        sol['pb_dn'][a][t] = a_dn[(a,t)].value()
        # no explicit s_up/s_dn in stochastic
    return sol
