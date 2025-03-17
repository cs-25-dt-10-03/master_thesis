from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpAffineExpression
from typing import List
from flexoffer_logic import DFO, findOrInterpolatePoints, DependencyPolygon, Point

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


def DFO_Optimization(dfo: DFO, cost_per_unit: list) -> list:
    """Optimizes the scheduling of energy usage for a given DFO based on cost per unit of energy, 
       ensuring that each step respects cumulative dependency constraints **and** per-hour constraints."""
    
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