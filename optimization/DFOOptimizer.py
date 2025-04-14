from pulp import LpProblem, LpMaximize, LpMinimize, LpVariable, lpSum, LpAffineExpression
from typing import List
from database.dataManager import get_price_at_datetime, fetch_mFRR_by_range
from config import config
from flexoffer_logic import DFO, findOrInterpolatePoints, DependencyPolygon, Point
import pandas as pd

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
    up_prices = prices_df['mFRR_UpPriceDKK'].str.replace(',', '.').astype(float).tolist()
    down_prices = prices_df['mFRR_DownPriceDKK'].str.replace(',', '.').astype(float).tolist()

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