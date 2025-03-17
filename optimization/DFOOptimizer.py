from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from typing import List
from flexoffer_logic import DFO, DependencyPolygon, Point


def add_linear_interpolation_constraints(model, energy, cumulative_dependency, points):
    """Adds linear interpolation constraints for energy allocation based on dependency polygons."""
    for k in range(1, len(points) - 1, 2):
        prev_point_min = points[k - 1]
        prev_point_max = points[k]
        next_point_min = points[k + 1]
        next_point_max = points[k + 2]

        if prev_point_min.x <= next_point_min.x:
            # Constraint: cumulative_dependency should be within this range
            model += cumulative_dependency >= prev_point_min.x
            model += cumulative_dependency <= next_point_min.x

            # Linear interpolation for min/max energy
            min_energy = (
                prev_point_min.y
                + ((next_point_min.y - prev_point_min.y) / (next_point_min.x - prev_point_min.x))
                * (cumulative_dependency - prev_point_min.x)
            )

            max_energy = (
                prev_point_max.y
                + ((next_point_max.y - prev_point_max.y) / (next_point_max.x - prev_point_max.x))
                * (cumulative_dependency - prev_point_max.x)
            )

            # Constraint: Ensure energy stays within the interpolated min/max
            model += energy >= min_energy
            model += energy <= max_energy

            break  # Stop after finding the first valid range


def DFO_Optimization(dfo: DFO, cost_per_unit: List[float]) -> List[float]:
    """
    Optimizes a DFO schedule using PuLP.
    
    Args:
        dfo (DFO): The dependency flex offer to optimize.
        cost_per_unit (List[float]): Cost per energy unit at each timestep.
    
    Returns:
        List[float]: The optimized schedule.
    """
    num_timesteps = len(dfo.polygons)
    if num_timesteps != len(cost_per_unit):
        raise ValueError("Mismatch between DFO timesteps and cost array length.")

    # Define the optimization model
    model = LpProblem("DFO_Scheduling", LpMinimize)

    # Decision variables: Energy allocation per timestep
    energy_alloc = [LpVariable(f"energy_{t}", lowBound=0, cat="Continuous") for t in range(num_timesteps)]

    # Objective function: Minimize total cost
    model += lpSum(cost_per_unit[t] * energy_alloc[t] for t in range(num_timesteps)), "Total_Cost"

    # Constraints: Ensure scheduling adheres to dependency polygons
    cumulative_dependency = 0  # Running sum of energy allocations
    for t in range(num_timesteps):
        polygon = dfo.polygons[t]
        
        # Add interpolation constraints for energy allocation within the dependency polygon
        add_linear_interpolation_constraints(model, energy_alloc[t], cumulative_dependency, polygon.points)

        # Update cumulative dependency for next timestep
        cumulative_dependency += energy_alloc[t]

    # Solve the optimization problem
    model.solve()

    # Extract results
    optimized_schedule = [energy_alloc[t].varValue for t in range(num_timesteps)]
    
    return optimized_schedule