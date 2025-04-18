from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import datetime, timedelta, timezone
from typing import List, Tuple, Dict
from database.dataManager import get_prices_in_range, fetch_mFRR_by_range, fetch_Regulating_by_range
import pandas as pd
import csv
import os
import numpy as np
import pulp
from config import config
import time
from flexoffer_logic import Flexoffer, TimeSlice



def optimize_spot_only(fos: List[Flexoffer]) -> Tuple[List[Flexoffer], Dict]:
    import pulp
    import time
    from datetime import timedelta
    import pandas as pd

    start_timer = time.time()
    delta_t = config.TIME_RESOLUTION / 3600.0

    est_global = min(fo.get_est() for fo in fos)
    et_global = max(fo.get_et() for fo in fos)
    est_global, et_global = pd.to_datetime(est_global, unit="s"), pd.to_datetime(et_global, unit="s")
    T = int((et_global - est_global).total_seconds() // config.TIME_RESOLUTION)
    time_index = [est_global + timedelta(seconds=config.TIME_RESOLUTION * t) for t in range(T)]

    spot_prices_df = get_prices_in_range(est_global, et_global).groupby(level=0).first()
    model = pulp.LpProblem("Spot_Only_Optimization", pulp.LpMinimize)
    vars_by_fo = {}
    obj_terms = []

    for fo in fos:
        i = fo.get_offer_id()
        d = fo.get_duration()
        profile = fo.get_profile()
        allowed_starts = fo.get_allowed_start_times()
        start_offsets = [(s - fo.get_est()) // config.TIME_RESOLUTION for s in allowed_starts]

        s_vars = {s: pulp.LpVariable(f"s_{i}_{s}", cat="Binary") for s in start_offsets}
        p_t = {t: pulp.LpVariable(f"p_{i}_{t}", lowBound=0) for t in range(T)}

        model += pulp.lpSum(s_vars.values()) == 1, f"one_start_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) >= fo.get_min_overall_alloc(), f"min_alloc_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) <= fo.get_max_overall_alloc(), f"max_alloc_{i}"

        for s, s_var in s_vars.items():
            for t_rel in range(d):
                t = s + t_rel
                if t >= T:
                    continue
                tslice = profile[t_rel]
                model += p_t[t] <= tslice.max_power * s_var, f"pmax_{i}_{t}"
                model += p_t[t] >= tslice.min_power * s_var, f"pmin_{i}_{t}"

        for t in range(T):
            if not any(s <= t < s + d for s in s_vars):
                model += p_t[t] == 0, f"inactive_{i}_{t}"

            ts = time_index[t]
            price = spot_prices_df.loc[ts, "spot_price_DKK"]
            obj_terms.append(p_t[t] * price)

        vars_by_fo[i] = (fo, p_t, s_vars)

    model += pulp.lpSum(obj_terms), "Total_Spot_Cost"
    model.solve()

    for i, (fo, p_t, s_vars) in vars_by_fo.items():
        chosen_starts = [s for s in s_vars if pulp.value(s_vars[s]) > 0.5]
        if not chosen_starts:
            continue
        start_index = max(chosen_starts)
        start_time = fo.get_est() + start_index * config.TIME_RESOLUTION
        fo.set_scheduled_start_time(start_time)
        allocation = [pulp.value(p_t[t]) for t in range(start_index, min(start_index + fo.get_duration(), T))]
        fo.set_scheduled_allocation(allocation)

    runtime = time.time() - start_timer
    return fos, {"strategy": "spot_only", "spot_cost": pulp.value(model.objective), "runtime": runtime}












def mFRR_reserve_only(flexoffers: List[Flexoffer]) -> List[Flexoffer]:
    results = []
    optimized_fos = []

    for FO in flexoffers:
        p_max = FO.get_profile()[0].max_power
        d = FO.get_duration()

        est, et = FO.get_est_hour(), FO.get_et_hour()
        T = et + 24 - est if et < est else et - est

        prices_df = fetch_mFRR_by_range(FO.get_est(), FO.get_et())
        up_prices = prices_df['mFRR_UpPriceDKK'].to_numpy()
        down_prices = prices_df['mFRR_DownPriceDKK'].to_numpy()

        model = pulp.LpProblem(f"FlexOffer_{FO.get_offer_id()}", pulp.LpMaximize)
        p_t, p_up, p_down = {}, {}, {}
        start_vars = {s: pulp.LpVariable(f"start_{s}", cat='Binary') for s in range(T - d + 1)}

        for t in range(T):
            p_t[t] = pulp.LpVariable(f"p_{t}", lowBound=0, upBound=p_max)
            p_up[t] = pulp.LpVariable(f"r_up_{t}", lowBound=0, upBound=p_max)
            p_down[t] = pulp.LpVariable(f"r_down_{t}", lowBound=0, upBound=p_max)

        model += pulp.lpSum(p_up[t] * up_prices[t] + p_down[t] * down_prices[t] for t in range(T))

        model += pulp.lpSum(start_vars.values()) == 1
        for t in range(T):
            model += p_down[t] <= p_t[t]
            model += p_up[t] <= p_max - p_t[t]

            valid_starts = [s for s in start_vars if s <= t < s + d]
            if valid_starts:
                model += p_t[t] <= p_max * pulp.lpSum(start_vars[s] for s in valid_starts)
            else:
                model += p_t[t] == 0

        model.solve()

        selected_start = next((s for s in start_vars if pulp.value(start_vars[s]) > 0.5), None)
        schedule = [pulp.value(p_t[t]) for t in range(selected_start, selected_start + d)] if selected_start is not None else []
        FO.set_scheduled_allocation(schedule)
        FO.set_scheduled_start_time(FO.get_est() + selected_start * 3600)
        optimized_fos.append(FO)

        # Logging results
        total_revenue = 0
        for t in range(T):
            pt = pulp.value(p_t[t])
            rup = pulp.value(p_up[t])
            rdown = pulp.value(p_down[t])
            up_price = up_prices[t]
            down_price = down_prices[t]

            total_revenue += (rup * up_price + rdown * down_price)

            results.append({
                'offer_id': FO.get_offer_id(),
                't': t,
                'charge_kW': pt,
                'r_up_kW': rup,
                'r_down_kW': rdown,
                'up_price_DKK': up_price,
                'down_price_DKK': down_price,
                'start_offset': selected_start,
                'duration': d,
                'total_revenue': None  # will be added only once per FO
            })

        if results:
            results[-1]['total_revenue'] = total_revenue

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation/results/results.csv", index=False)
    return optimized_fos























def optimize_Spot_and_mfrr(fos: List[Flexoffer]) -> Tuple[List[Flexoffer], Dict]:

    start_timer = time.time()
    delta_t = config.TIME_RESOLUTION / 3600.0

    # --- Time range ---
    est_global = min(FO.get_est() for FO in fos)
    et_global = max(FO.get_et() for FO in fos)
    est_global, et_global = pd.to_datetime(est_global, unit='s'), pd.to_datetime(et_global, unit='s')

    print(f"to make sure {est_global}")

    T = int((et_global - est_global).total_seconds() // config.TIME_RESOLUTION)
    time_index = [est_global + timedelta(seconds=config.TIME_RESOLUTION * t) for t in range(T)]
    
    for t in range(T):
        print(time_index[t])

    # --- Prices ---
    spot_prices_df = get_prices_in_range(est_global, et_global)
    mFRR_reserve_df = fetch_mFRR_by_range(est_global, et_global)

    # --- Optimization ---
    vars_by_fo = {}
    obj_terms = []
    model = pulp.LpProblem("Joint_spot_reserve", pulp.LpMaximize)

    for FO in fos:
        i = FO.get_offer_id()
        d = FO.get_duration()
        profile = FO.get_profile()
        allowed_starts = FO.get_allowed_start_times()
        start_offsets = [(s - FO.get_est()) // config.TIME_RESOLUTION for s in allowed_starts]

        # Start time variables
        s_vars = {s: pulp.LpVariable(f"s_{i}_{s}", cat='Binary') for s in start_offsets}

        # Power & reserve variables
        p_t, r_up, r_down = {}, {}, {}
        for t in range(T):
            p_t[t] = pulp.LpVariable(f"p_{i}_{t}", lowBound=0)
            r_up[t] = pulp.LpVariable(f"r_up_{i}_{t}", lowBound=0)
            r_down[t] = pulp.LpVariable(f"r_down_{i}_{t}", lowBound=0)

        # Energy constraints         
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) >= FO.get_min_overall_alloc(), f"min__alloc_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) <= FO.get_max_overall_alloc(), f"max__alloc_{i}"

        for t in range(T):
            # Reserve feasibility
            model += r_up[t] <= p_t[t], f"up_bound_{i}_{t}"
            valid_starts = [s for s in s_vars if s <= t < s + d]
            if valid_starts:
                t_rel = [t - s for s in valid_starts if (t - s) < d]
                if t_rel:
                    tr = t_rel[0]
                    max_power = profile[tr].max_power
                    min_power = profile[tr].min_power
                    act = pulp.lpSum(s_vars[s] for s in valid_starts)
                    model += p_t[t] <= max_power * act, f"active_upper_{i}_{t}"
                    model += p_t[t] >= min_power * act, f"active_lower_{i}_{t}"
                    model += r_down[t] <= max_power * act - p_t[t], f"down_bound_{i}_{t}"
            else:
                model += p_t[t] == 0, f"inactive_{i}_{t}"

            ts = time_index[t]
            spot_price = spot_prices_df.loc[ts, 'SpotPriceDKK']
            up_price = mFRR_reserve_df.loc[ts, 'mFRR_UpPriceDKK']
            down_price = mFRR_reserve_df.loc[ts, 'mFRR_DownPriceDKK']

            obj_terms.append(r_up[t] * up_price + r_down[t] * down_price - p_t[t] * spot_price)

        model += pulp.lpSum(s_vars.values()) == 1, f"one_start_{i}"
        vars_by_fo[i] = (FO, p_t, r_up, r_down, s_vars)

    model += pulp.lpSum(obj_terms), "Total_Revenue"
    model.solve()


    #### OUTPUTTING AND LOGGING RESULT #####
    total_revenue = 0.0
    for i, (FO, p_t, r_up, r_down, s_vars) in vars_by_fo.items():
        chosen_starts = [s for s in s_vars if pulp.value(s_vars[s]) > 0.5]
        if not chosen_starts:
            print(f"FlexOffer {i} was not scheduled (no valid start time selected)")
            continue  # Skip this FO to avoid crash
        start_index = max(chosen_starts)



        start_time = FO.get_est() + start_index * config.TIME_RESOLUTION
        FO.set_scheduled_start_time(start_time)

        allocation = [
            pulp.value(p_t[t])
            for t in range(start_index, min(start_index + FO.get_duration(), T))
        ]
        FO.set_scheduled_allocation(allocation)

        # Sum up the revenue for this flexOffer over its active duration
        for j in range(min(FO.get_duration(), T - start_index)):
            t = start_index + j
            ts = time_index[t]
            spot_price = spot_prices_df.loc[ts, 'SpotPriceDKK']
            up_price = mFRR_reserve_df.loc[ts, 'mFRR_UpPriceDKK']
            down_price = mFRR_reserve_df.loc[ts, 'mFRR_DownPriceDKK']
            total_revenue += (pulp.value(r_up[t]) * up_price + pulp.value(r_down[t]) * down_price - pulp.value(p_t[t]) * spot_price)

    runtime = time.time() - start_timer




    # --- Logging
    output_path = "optimization_schedule_log.csv"
    with open(output_path, mode='w', newline='') as file:
        import csv
        writer = csv.writer(file)
        writer.writerow(["FlexOfferID", "Timestamp", "ScheduledPower_kW", "ReserveUp_kW", "ReserveDown_kW"])
        for i, (FO, p_t, r_up, r_down, s_vars) in vars_by_fo.items():
            chosen_starts = [s for s in s_vars if pulp.value(s_vars[s]) > 0.5]
            if not chosen_starts:
                continue  # Skip this FO to avoid crash
            start_index = max(chosen_starts)
            start_time = FO.get_est() + start_index * config.TIME_RESOLUTION
            duration = FO.get_duration()
            for j in range(duration):
                t = start_index + j
                if t in p_t:
                    ts = pd.to_datetime(start_time + j * config.TIME_RESOLUTION, unit='s')
                    writer.writerow([
                        i,
                        ts,
                        pulp.value(p_t[t]),
                        pulp.value(r_up[t]),
                        pulp.value(r_down[t])
                    ])

    results_dict = {
        "revenue": total_revenue,
        "runtime": runtime
    }
    return fos, results_dict
























def sequential_schedule_mfrr_then_spot(fos: List[Flexoffer]) -> List[Flexoffer]:
    start_timer = time.time()
    delta_t = config.TIME_RESOLUTION / 3600.0

    # --- Time range ---
    est_global = min(FO.get_est() for FO in fos)
    et_global = max(FO.get_lst() + FO.get_duration() * config.TIME_RESOLUTION for FO in fos)
    est_global, et_global = pd.to_datetime(est_global, unit='s'), pd.to_datetime(et_global, unit='s')
    T = int((et_global - est_global).total_seconds() // config.TIME_RESOLUTION)
    time_index = [est_global + timedelta(seconds=config.TIME_RESOLUTION * t) for t in range(T)]

    # --- Prices ---
    spot_prices_df = get_prices_in_range(est_global, et_global)
    mFRR_reserve_df = fetch_mFRR_by_range(est_global, et_global)

    # === Stage 1: mFRR Reserve Capacity Scheduling ===
    model = pulp.LpProblem("Sequential_mFRR_Only", pulp.LpMaximize)
    vars_by_fo = {}
    mfrr_obj_terms = []

    for FO in fos:
        i = FO.get_offer_id()
        d = FO.get_duration()
        profile = FO.get_profile()
        allowed_starts = FO.get_allowed_start_times()
        start_offsets = [(s - FO.get_est()) // config.TIME_RESOLUTION for s in allowed_starts]

        s_vars = {s: pulp.LpVariable(f"s_{i}_{s}", cat='Binary') for s in start_offsets}
        model += pulp.lpSum(s_vars.values()) == 1, f"one_start_{i}"

        p_t, r_up, r_down = {}, {}, {}
        for t in range(T):
            p_t[t] = pulp.LpVariable(f"p_{i}_{t}", lowBound=0)
            r_up[t] = pulp.LpVariable(f"r_up_{i}_{t}", lowBound=0)
            r_down[t] = pulp.LpVariable(f"r_down_{i}_{t}", lowBound=0)

        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) >= FO.get_min_overall_alloc(), f"min__alloc_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) <= FO.get_max_overall_alloc(), f"max__alloc_{i}"

        for t in range(T):
            model += r_up[t] <= p_t[t], f"up_bound_{i}_{t}"
            valid_starts = [s for s in s_vars if s <= t < s + d]
            if valid_starts:
                t_rel = [t - s for s in valid_starts if (t - s) < d]
                if t_rel:
                    tr = t_rel[0]
                    max_power = profile[tr].max_power
                    min_power = profile[tr].min_power
                    act = pulp.lpSum(s_vars[s] for s in valid_starts)
                    model += p_t[t] <= max_power * act, f"active_upper_{i}_{t}"
                    model += p_t[t] >= min_power * act, f"active_lower_{i}_{t}"
                    model += r_down[t] <= max_power * act - p_t[t], f"down_bound_{i}_{t}"
            else:
                model += p_t[t] == 0, f"inactive_{i}_{t}"

            ts = time_index[t]
            up_price = mFRR_reserve_df.loc[ts, 'mFRR_UpPriceDKK']
            down_price = mFRR_reserve_df.loc[ts, 'mFRR_DownPriceDKK']
            mfrr_obj_terms.append(r_up[t] * up_price + r_down[t] * down_price)

        vars_by_fo[i] = (FO, p_t, r_up, r_down, s_vars)

    model += pulp.lpSum(mfrr_obj_terms), "mFRR_Revenue_Only"
    model.solve()

    # --- Apply the mFRR schedule ---
    mfrr_revenue = 0.0
    for i, (FO, p_t, r_up, r_down, s_vars) in vars_by_fo.items():

        chosen_starts = [s for s in s_vars if pulp.value(s_vars[s]) > 0.5]
        if not chosen_starts:
            print(f"⚠️ FlexOffer {i} was not scheduled (no valid start time selected)")
            continue  # Skip this FO to avoid crash
        start_index = max(chosen_starts)


        start_time = FO.get_est() + start_index * config.TIME_RESOLUTION
        FO.set_scheduled_start_time(start_time)

        allocation = []
        for j in range(FO.get_duration()):
            t = start_index + j
            val = pulp.value(p_t[t])
            allocation.append(val)

            ts = pd.to_datetime(start_time + j * config.TIME_RESOLUTION, unit='s')
            mfrr_revenue += (
                pulp.value(r_up[t]) * mFRR_reserve_df.loc[ts, 'mFRR_UpPriceDKK'] +
                pulp.value(r_down[t]) * mFRR_reserve_df.loc[ts, 'mFRR_DownPriceDKK']
            )

        FO.set_scheduled_allocation(allocation)

    # === Stage 2: Spot Market Cost Calculation ===
    spot_cost = 0.0
    for FO in fos:
        start = FO.get_scheduled_start_time()
        for j, val in enumerate(FO.get_scheduled_allocation()):
            ts = pd.to_datetime(start + j * config.TIME_RESOLUTION, unit='s')
            spot_cost += val * spot_prices_df.loc[ts, 'SpotPriceDKK'] * delta_t

    total_revenue = mfrr_revenue - spot_cost
    runtime = time.time() - start_timer

    return fos, {
        "revenue": total_revenue,
        "mfrr_revenue": mfrr_revenue,
        "spot_cost": spot_cost,
        "runtime": runtime
    }

def optimize_spot_reserve_activation(fos: List[Flexoffer],
                                        delta_up: Dict[int, int],
                                        delta_down: Dict[int, int]) -> Tuple[List[Flexoffer], Dict]:
    import pulp
    import time
    from datetime import timedelta
    import pandas as pd

    start_timer = time.time()
    delta_t = config.TIME_RESOLUTION / 3600.0

    est_global = min(FO.get_est() for FO in fos)
    et_global = max(FO.get_et() for FO in fos)
    est_global, et_global = pd.to_datetime(est_global, unit='s'), pd.to_datetime(et_global, unit='s')
    T = int((et_global - est_global).total_seconds() // config.TIME_RESOLUTION)
    time_index = [est_global + timedelta(seconds=config.TIME_RESOLUTION * t) for t in range(T)]

    spot_prices_df = get_prices_in_range(est_global, et_global).groupby(level=0).first()
    mFRR_reserve_df = fetch_mFRR_by_range(est_global, et_global).groupby(level=0).first()

    model = pulp.LpProblem("Joint_with_Activation", pulp.LpMaximize)
    vars_by_fo = {}
    obj_terms = []

    penalty = 1000  # Slack penalty (π_p)

    # Activation and slack vars
    p_act_up, p_act_down, s_up, s_down = {}, {}, {}, {}
    for t in range(T):
        p_act_up[t] = pulp.LpVariable(f"p_act_up_{t}", lowBound=0)
        p_act_down[t] = pulp.LpVariable(f"p_act_down_{t}", lowBound=0)
        s_up[t] = pulp.LpVariable(f"s_up_{t}", lowBound=0)
        s_down[t] = pulp.LpVariable(f"s_down_{t}", lowBound=0)

    for FO in fos:
        i = FO.get_offer_id()
        d = FO.get_duration()
        profile = FO.get_profile()
        allowed_starts = FO.get_allowed_start_times()
        start_offsets = [(s - FO.get_est()) // config.TIME_RESOLUTION for s in allowed_starts]

        s_vars = {s: pulp.LpVariable(f"s_{i}_{s}", cat='Binary') for s in start_offsets}
        p_t, r_up, r_down = {}, {}, {}
        for t in range(T):
            p_t[t] = pulp.LpVariable(f"p_{i}_{t}", lowBound=0)
            r_up[t] = pulp.LpVariable(f"r_up_{i}_{t}", lowBound=0)
            r_down[t] = pulp.LpVariable(f"r_down_{i}_{t}", lowBound=0)

        model += pulp.lpSum(s_vars.values()) == 1, f"one_start_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) >= FO.get_min_overall_alloc(), f"min_alloc_{i}"
        model += pulp.lpSum(p_t[t] * delta_t for t in range(T)) <= FO.get_max_overall_alloc(), f"max_alloc_{i}"

        # Enforce activation window and profile bounds using direct (s, t_rel) indexing
        for s, s_var in s_vars.items():
            for t_rel in range(d):
                t = s + t_rel
                if t >= T:
                    continue
                tslice = profile[t_rel]
                model += p_t[t] <= tslice.max_power * s_var, f"p_upper_{i}_{t}"
                model += p_t[t] >= tslice.min_power * s_var, f"p_lower_{i}_{t}"
                model += r_up[t] <= p_t[t], f"up_feas_{i}_{t}"
                model += r_down[t] <= tslice.max_power * s_var - p_t[t], f"down_feas_{i}_{t}"

        for t in range(T):
            if not any(s <= t < s + d for s in s_vars):
                model += p_t[t] == 0, f"inactive_{i}_{t}"

            ts = time_index[t]
            spot_price = spot_prices_df.loc[ts, 'SpotPriceDKK']
            up_price = mFRR_reserve_df.loc[ts, 'mFRR_UpPriceDKK']
            down_price = mFRR_reserve_df.loc[ts, 'mFRR_DownPriceDKK']
            act_up_price = mFRR_reserve_df.loc[ts, 'BalancingPowerPriceUpDKK']
            act_down_price = mFRR_reserve_df.loc[ts, 'BalancingPowerPriceDownDKK']

            obj_terms.append(
                r_up[t] * up_price + r_down[t] * down_price - p_t[t] * spot_price +
                p_act_up[t] * act_up_price + p_act_down[t] * act_down_price -
                penalty * (s_up[t] + s_down[t])
            )

            model += p_act_up[t] + s_up[t] >= r_up[t] * delta_up.get(t, 0), f"act_up_req_{t}"
            model += p_act_down[t] + s_down[t] >= r_down[t] * delta_down.get(t, 0), f"act_down_req_{t}"

        vars_by_fo[i] = (FO, p_t, r_up, r_down, s_vars)

    model += pulp.lpSum(obj_terms), "Total_Revenue"
    model.solve()

    for i, (FO, p_t, _, _, s_vars) in vars_by_fo.items():
        chosen_starts = [s for s in s_vars if pulp.value(s_vars[s]) > 0.5]
        if not chosen_starts:
            continue
        start_index = max(chosen_starts)
        start_time = FO.get_est() + start_index * config.TIME_RESOLUTION
        FO.set_scheduled_start_time(start_time)
        allocation = [pulp.value(p_t[t]) for t in range(start_index, min(start_index + FO.get_duration(), T))]
        FO.set_scheduled_allocation(allocation)

    return fos, {"status": pulp.LpStatus[model.status], "objective": pulp.value(model.objective)}


def predict_deltas_from_spot(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[Dict[int, int], Dict[int, int]]:
    import pandas as pd
    from database.dataManager import get_prices_in_range, fetch_mFRR_by_range, fetch_regulation_by_range

    # Load price data
    spot_df = get_prices_in_range(start, end)
    mfrr_df = fetch_mFRR_by_range(start, end)
    reg_df = fetch_Regulating_by_range(start, end)

    # Align by timestamp
    merged = spot_df.join([mfrr_df, reg_df], how="inner")
    merged = merged.dropna(subset=["spot_price_DKK", "mFRR_UpPriceDKK", "mFRR_DownPriceDKK",
                                   "BalancingPowerPriceUpDKK", "BalancingPowerPriceDownDKK"])

    # Bid policy: linear in spot
    a_up, b_up = 1.5, 0.015
    a_down, b_down = 0.5, 0.010
    merged["UpBid"] = a_up + b_up * merged["spot_price_DKK"]
    merged["DownBid"] = a_down + b_down * merged["spot_price_DKK"]

    # Apply activation logic
    merged["delta_up"] = (merged["UpBid"] < merged["BalancingPowerPriceUpDKK"]).astype(int)
    merged["delta_down"] = (merged["DownBid"] < merged["BalancingPowerPriceDownDKK"]).astype(int)

    delta_up_dict = dict(enumerate(merged["delta_up"].values))
    delta_down_dict = dict(enumerate(merged["delta_down"].values))
    return delta_up_dict, delta_down_dict
