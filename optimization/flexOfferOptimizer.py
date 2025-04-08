from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import datetime, timedelta, timezone
from typing import List
from database.dataManager import get_prices_in_range, fetch_mFRR_by_range, fetch_Regulating_by_range
import pandas as pd
import numpy as np
import pulp
from config import config
from flexoffer_logic import Flexoffer, TimeSlice

def optimize(fos : List[Flexoffer]) -> Flexoffer:

    model = pulp.LpProblem("FlexOffer_scheduling", pulp.LpMinimize)

    all_vars = {}
    total_cost_terms = []
    
    for idx, FO in enumerate(fos):
        est = FO.get_est_hour()
        et = FO.get_et_hour()
        duration = FO.get_duration()
        allowed_starts = FO.get_allowed_start_times()
        profile = FO.get_profile()

        # spot prices for this FO’s time window
        prices = get_prices_in_range(FO.get_est(), FO.get_et())

        # start time binary vars
        start_vars = {s: pulp.LpVariable(f"{fo_id}_start_{s}", cat="Binary") for s in allowed_starts}

        # enforce exactly one start time
        model += pulp.lpSum(start_vars[s] for s in allowed_starts) == 1, f"{fo_id}_only_one_start"

        # decision variables: power at each local time slice (0..duration)
        power_vars = [
            pulp.LpVariable(f"{fo_id}_power_{t}", lowBound=profile[t].min_power, upBound=profile[t].max_power)
            for t in range(duration)
        ]

        #enforce min/max total energy
        total_energy = pulp.lpSum(power_vars)
        if FO.get_min_overall_alloc() > 0:
            model += total_energy >= FO.get_min_overall_alloc(), f"{fo_id}_min_energy"
            model += total_energy <= FO.get_max_overall_alloc(), f"{fo_id}_max_energy"

        # add to global cost terms, not sure about this but whatever lmao
        for t in range(duration):
            global_t = (min(allowed_starts) + t) % 24
            total_cost_terms.append(power_vars[t] * prices[t])

        # store vars for later retrieval
        all_vars[fo_id] = {
            "power_vars": power_vars,
            "start_vars": start_vars,
            "fo": FO
        }

    # OBJECTIVE: total cost across all flex offers
    model += pulp.lpSum(total_cost_terms), "Total_Cost"

    # Solve the model
    model.solve()

    # Process results and update FlexOffers
    for fo_id, vars in all_vars.items():
        power_values = [v.varValue for v in vars["power_vars"]]
        start_time = next((s for s, v in vars["start_vars"].items() if v.varValue == 1), None)
        fo = vars["fo"]
        fo.set_scheduled_allocation(power_values)
        fo.set_scheduled_start_time(start_time)

    return [v["fo"] for v in all_vars.values()]



def MultiMarketoptimize(flexoffers: list[Flexoffer]) -> tuple[list[Flexoffer], pd.DataFrame]:
    results = []
    optimized_fos = []

    for FO in flexoffers:
        p_max = FO.get_profile()[0].max_power
        d = FO.get_duration()

        est, et = FO.get_est_hour(), FO.get_et_hour()
        T = et + 24 - est if et < est else et - est

        prices_df = fetch_mFRR_by_range(FO.get_est(), FO.get_et())
        up_prices = prices_df['mFRR_UpPriceDKK'].str.replace(',', '.').astype(float).to_numpy()
        down_prices = prices_df['mFRR_DownPriceDKK'].str.replace(',', '.').astype(float).to_numpy()

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
            model += p_up[t] <= p_t[t]
            model += p_down[t] <= p_max - p_t[t]

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
