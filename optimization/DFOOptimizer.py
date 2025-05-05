from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpAffineExpression
from typing import List, Tuple, Dict
from database.dataManager import  get_price_at_datetime, get_prices_in_range, fetch_mFRR_by_range, fetch_Regulating_by_range, load_and_prepare_prices
from config import config
from flexoffer_logic import DFO, findOrInterpolatePoints, padDFOsToCommonTimeline, DependencyPolygon, Point
import pandas as pd
import pulp
from config import config

def get_cost_array_for_dfo(dfo) -> list[float]:
    num_timesteps = len(dfo.polygons)
    cost_array = []
    for t in range(num_timesteps):
        step_timestamp = dfo.earliest_start_time + t * config.TIME_RESOLUTION
        price = get_price_at_datetime(step_timestamp)
        cost_array.append(price)
    return cost_array


def add_interpolation_constraints(model, energy_var, cumulative_energy_var, points):
    """Applies linear interpolation constraints based on dependency values in a polygon."""
    
    for k in range(1, len(points) - 1, 2):  # Iterate through pairs of min/max dependency values
        prev_point_min = points[k - 1]
        prev_point_max = points[k]
        next_point_min = points[k + 1]
        next_point_max = points[k + 2]

        # Ensure dependency range is within bounds
        if prev_point_min.x <= cumulative_energy_var <= next_point_min.x:

            # Compute interpolated min/max energy using linear interpolation
            if next_point_min.x != prev_point_min.x:  # Avoid division by zero
                min_energy = prev_point_min.y + ((next_point_min.y - prev_point_min.y) /
                                                 (next_point_min.x - prev_point_min.x)) * (cumulative_energy_var - prev_point_min.x)

                max_energy = prev_point_max.y + ((next_point_max.y - prev_point_max.y) /
                                                 (next_point_max.x - prev_point_max.x)) * (cumulative_energy_var - prev_point_max.x)

                # Enforce energy bounds for this dependency range
                model += energy_var >= min_energy
                model += energy_var <= max_energy

                break  # Stop after applying constraints for the relevant range


def DFO_Optimization(dfo: DFO) -> list:
    """Optimizes the scheduling of energy usage for a given DFO based on cost per unit of energy, 
       ensuring that each step respects cumulative dependency constraints **and** per-hour constraints."""

    cost_per_unit = get_cost_array_for_dfo(dfo)
    num_timesteps = len(dfo.polygons)
    if num_timesteps != len(cost_per_unit):
        raise ValueError("Mismatch between DFO timesteps and cost_per_unit size.")

    # Define LP problem
    model = LpProblem("DFO_Optimization", LpMinimize)

    # Decision variables: Energy allocation per timestep
    energy_alloc = [LpVariable(f"energy_{t}", lowBound=0, upBound=None, cat="Continuous") for t in range(num_timesteps)]

    # Objective function: Minimize total energy cost (prioritizes charging in low-cost hours)
    model += lpSum(cost_per_unit[t] * energy_alloc[t] for t in range(num_timesteps)), "Total_Cost"

    # Cumulative energy usage (dependency tracking)
    cumulative_energy = 0  # Start with 0 cumulative energy

    # Constraints: Ensure each timestep respects its **own dependency polygon constraints**
    for t in range(num_timesteps):
        polygon = dfo.polygons[t]

        # Ensure energy allocation is within the allowed range for this specific time step
        energy_min = min(p.y for p in polygon.points)  # Min energy possible for this hour
        energy_max = max(p.y for p in polygon.points)  # Max energy possible for this hour
        model += energy_alloc[t] >= energy_min
        model += energy_alloc[t] <= energy_max

        # Apply **linear interpolation constraints** for dependency tracking
        add_interpolation_constraints(model, energy_alloc[t], cumulative_energy, polygon.points)

        # **Update cumulative dependency correctly**
        cumulative_energy += energy_alloc[t]  # Keep a running sum

    # Solve the problem
    model.solve()

    # Retrieve optimized energy allocations
    optimized_schedule = [p.varValue for p in energy_alloc]

    return optimized_schedule

# 🧠 : Optimize DFO to mazimize mFFR revenue
def DFO_MultiMarketOptimization(dfo: DFO) -> pd.DataFrame:
    num_timesteps = len(dfo.polygons)

    # Fetch mFRR prices for the given time range
    prices_df = fetch_mFRR_by_range(dfo.get_est(), dfo.get_et())
    up_prices = pd.to_numeric(prices_df['mFRR_UpPriceDKK'], errors='coerce').fillna(0).tolist()
    down_prices = pd.to_numeric(prices_df['mFRR_DownPriceDKK'], errors='coerce').fillna(0).tolist()

    # Ensure length matches
    if len(up_prices) < num_timesteps or len(down_prices) < num_timesteps:
        raise ValueError(f"Price data too short: {len(up_prices)=}, {len(down_prices)=}, {num_timesteps=}")

    model = LpProblem("DFO_MultiMarket", LpMaximize)

    # Decision variables: 
    # p_t: power usage from spotmarket
    # r_up: Power reserve for mFRR upward regulation
    # r_down: Power reserve for mFRR downward regulation
    p_t = [LpVariable(f"p_{t}", lowBound=0) for t in range(num_timesteps)]
    r_up = [LpVariable(f"r_up_{t}", lowBound=0) for t in range(num_timesteps)]
    r_down = [LpVariable(f"r_down_{t}", lowBound=0) for t in range(num_timesteps)]

    # Objective function: Maximize total revenue from mFRR services
    model += lpSum(r_up[t] * up_prices[t] + r_down[t] * down_prices[t] for t in range(num_timesteps)), "Total_Revenue"

    # Add DFO specific constraints
    cumulative_energy = LpAffineExpression() # 🧠 IMPORTANT: make cumulative_energy a pulp symbolic expression
    max_energy_expr = 0
    min_energy_expr = 0
    for t in range(num_timesteps):
        polygon = dfo.polygons[t]
        points = polygon.points

        if len(points) < 4:
            min_energy_expr = points[0].y
            max_energy_expr = points[1].y

            model += p_t[t] >= min_energy_expr, f"min_energy_{t}"
            model += p_t[t] <= max_energy_expr, f"max_energy_{t}"
            model += r_up[t] <= p_t[t], f"r_up_limit_{t}"
            model += r_down[t] <= max_energy_expr - p_t[t], f"r_down_limit_{t}"
        else:
            for k in range(1, len(points) - 2, 2):
                prev_min = points[k - 1]
                prev_max = points[k]
                next_min = points[k + 1]
                next_max = points[k + 2]
                
                if next_min.x == prev_min.x or next_max.x == prev_max.x:
                    continue
                
                min_energy_expr = prev_min.y + ((next_min.y - prev_min.y) / (next_min.x - prev_min.x)) * (cumulative_energy - prev_min.x)
                max_energy_expr = prev_max.y + ((next_max.y - prev_max.y) / (next_max.x - prev_max.x)) * (cumulative_energy - prev_max.x)
                
                model += p_t[t] >= min_energy_expr, f"min_interp_{t}"
                model += p_t[t] <= max_energy_expr, f"max_interp_{t}"
                break

            model += r_up[t] <= p_t[t], f"r_up_limit_{t}"
            model += r_down[t] <= max_energy_expr - p_t[t], f"r_down_limit_{t}"

        cumulative_energy += p_t[t]

    model.solve()

    results = []
    total_revenue = 0
    for t in range(num_timesteps):
        pt = p_t[t].varValue
        rup = r_up[t].varValue
        rdown = r_down[t].varValue
        up_price = up_prices[t]
        down_price = down_prices[t]
        total_revenue += (rup * up_price + rdown * down_price)

        results.append({
            'offer_id': dfo.dfo_id,
            't': t,
            'charge_kW': pt,
            'r_up_kW': rup,
            'r_down_kW': rdown,
            'up_price_DKK': up_price,
            'down_price_DKK': down_price,
            'total_revenue': None
        })

    if results:
        results[-1]['total_revenue'] = total_revenue

    df = pd.DataFrame(results)
    df.to_csv("evaluation/results/dfo_multimarket_results.csv", index=False)

    return df





def optimize_dfos(dfos: List[DFO]):
    """
    🧠 Optimizes DFOs across spot, reserve, and activation markets, based on config settings.
    Dynamically build LP alt efter config indstillinger.
    offers: list of (aggregated) DFOs
    spot_prices: Dataframe: [HourDK, SpotPriceDKK]
    reserve_prices: Dataframe: [HourDK, mFRR_UpPriceDKK, mFRR_DownPriceDKK]
    activation_prices: Dataframe: [HourDK, UpBalancingPriceDKK, DownBalancingPriceDKK]
    """

    # Align DFOs to a common timeline
    padded_dfos, T = padDFOsToCommonTimeline(dfos)
    A = len(padded_dfos)
    earliest_start = min(dfo.get_est() for dfo in padded_dfos)
    print(earliest_start)
    pd_earliest_start = pd.to_datetime(earliest_start, unit='s')

    # Load price data
    spot_prices, reserve_prices, activation_prices, indicators = load_and_prepare_prices(pd_earliest_start, T, resolution=config.TIME_RESOLUTION)

    def build_and_solve(use_spot, use_reserve, use_activation, fixed_p=None):
        prob = LpProblem("DFO_scheduling", LpMaximize)

        # Decision variables
        p_t = {(a, t): LpVariable(f"p_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_spot else None
        pr_up = {(a, t): LpVariable(f"pr_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_reserve else None
        pr_dn = {(a, t): LpVariable(f"pr_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_reserve else None
        pb_up = {(a, t): LpVariable(f"pb_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_activation else None
        pb_dn = {(a, t): LpVariable(f"pb_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_activation else None
        s_up = {(a, t): LpVariable(f"s_up_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_activation else None
        s_dn = {(a, t): LpVariable(f"s_dn_{a}_{t}", lowBound=0) for a in range(A) for t in range(T)} if use_activation else None

        # Objective function
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
                    obj.append(r_up * pr_up[(a, t)] + r_dn * pr_dn[(a, t)])
                if use_activation:
                    obj.append(b_up * pb_up[(a, t)] + b_dn * pb_dn[(a, t)])
                if use_spot:
                    dt = config.TIME_RESOLUTION / 3600.0
                    obj.append(-spot * p_t[(a, t)] * dt)
                if use_activation:
                    obj.append(- config.PENALTY * (s_up[(a, t)] + s_dn[(a, t)]))

        prob += lpSum(obj)

        # Constraints
        for a in range(A):
            cumulative_energy = LpAffineExpression() # 🧠 IMPORTANT: make cumulative_energy a pulp symbolic expression
            for t in range(T):
                polygon = padded_dfos[a].polygons[t]
                points = polygon.points

                # Interpolation constraints if polygon has multiple segments
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

                if use_spot:
                    prob += p_t[(a, t)] >= energy_min, f"min_energy_{a}_{t}"
                    prob += p_t[(a, t)] <= energy_max, f"max_energy_{a}_{t}"

                if use_reserve:
                    if use_spot:
                        prob += pr_up[(a, t)] <= p_t[(a, t)], f"r_up_limit_{a}_{t}"
                        prob += pr_dn[(a, t)] <= energy_max - p_t[(a, t)], f"r_dn_limit_{a}_{t}"
                    else:
                        prob += pr_up[(a, t)] <= energy_max, f"r_up_max_{a}_{t}"
                        prob += pr_dn[(a, t)] <= energy_max, f"r_dn_max_{a}_{t}"

                if use_activation:
                    delta_up, delta_dn = indicators[t]
                    prob += pb_up[(a, t)] + s_up[(a, t)] >= pr_up[(a, t)] * delta_up
                    prob += pb_dn[(a, t)] + s_dn[(a, t)] >= pr_dn[(a, t)] * delta_dn

                if use_spot:
                    cumulative_energy += p_t[(a, t)]

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # --- Extract solution ---
        solutions = {"p": {}, "pr_up": {}, "pr_dn": {}, "pb_up": {}, "pb_dn": {}, "s_up": {}, "s_dn": {}}
        for a in range(A):
            if use_spot:
                solutions["p"][a] = {t: pulp.value(p_t[(a, t)]) for t in range(T)}
            if use_reserve:
                solutions["pr_up"][a] = {t: pulp.value(pr_up[(a, t)]) for t in range(T)}
                solutions["pr_dn"][a] = {t: pulp.value(pr_dn[(a, t)]) for t in range(T)}
            if use_activation:
                solutions["pb_up"][a] = {t: pulp.value(pb_up[(a, t)]) for t in range(T)}
                solutions["pb_dn"][a] = {t: pulp.value(pb_dn[(a, t)]) for t in range(T)}
                solutions["s_up"][a] = {t: pulp.value(s_up[(a, t)]) for t in range(T)}
                solutions["s_dn"][a] = {t: pulp.value(s_dn[(a, t)]) for t in range(T)}

        # Set allocations back to DFOs
        for a, dfo in enumerate(padded_dfos):
            if use_spot:
                alloc = [solutions["p"][a][t] for t in range(T)]
                dfo.set_scheduled_allocation(alloc)

                # Determine scheduled start time
                try:
                    first_active = next(t for t, v in enumerate(alloc) if v > 1e-6)
                except StopIteration:
                    first_active = 0
                dfo.set_scheduled_start_time(earliest_start + first_active * config.TIME_RESOLUTION)

        return solutions

    # Joint vs Sequential optimization
    if config.MODE == "joint":
        return build_and_solve(
            use_spot=config.RUN_SPOT,
            use_reserve=config.RUN_RESERVE,
            use_activation=config.RUN_ACTIVATION
        )
    elif config.MODE == "sequential":
        sol1 = build_and_solve(
            use_spot=config.RUN_SPOT,
            use_reserve=config.RUN_RESERVE,
            use_activation=False
        )
        sol2 = build_and_solve(
            use_spot=False,
            use_reserve=config.RUN_RESERVE,
            use_activation=config.RUN_ACTIVATION
        )
        merged = sol1.copy()
        for key in ["pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn"]:
            merged[key] = sol2[key]
        return merged
    else:
        raise ValueError("Unknown MODE selected in config")