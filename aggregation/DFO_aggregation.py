from typing import List
from classes.DFO import Point, DependencyPolygon, DFO

def find_or_interpolate_points(points: List[Point], dependency_value: float) -> List[Point]:
    """Finds or interpolates points for a given dependency value."""
    matching_points = [point for point in points if point.x == dependency_value]

    if not matching_points:
        # Perform linear interpolation if an exact match is not found
        for k in range(1, len(points) - 1, 2):
            prev_point_min = points[k - 1]
            prev_point_max = points[k]
            next_point_min = points[k + 1]
            next_point_max = points[k + 2]

            if prev_point_min.x <= dependency_value <= next_point_min.x:
                s_min = linear_interpolation(
                    dependency_value, prev_point_min.x, prev_point_min.y, next_point_min.x, next_point_min.y
                )
                s_max = linear_interpolation(
                    dependency_value, prev_point_max.x, prev_point_max.y, next_point_max.x, next_point_max.y
                )
                return [Point(dependency_value, s_min), Point(dependency_value, s_max)]

    return matching_points

def linear_interpolation(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    """Performs linear interpolation."""
    if x1 == x0:  # Prevent division by zero
        return (y0 + y1) / 2
    return y0 + ((y1 - y0) * (x - x0) / (x1 - x0))

def agg2to1(dfo1: DFO, dfo2: DFO, numsamples: int) -> DFO:
    """Aggregates two DFOs into one."""
    if len(dfo1.polygons) != len(dfo2.polygons):
        raise RuntimeError("DFOs must have the same number of timesteps to aggregate. Kind Regards, agg2to1 function")

    aggregated_polygons = []

    for i in range(len(dfo1.polygons)):
        polygon1 = dfo1.polygons[i]
        polygon2 = dfo2.polygons[i]

        aggregated_min_prev = polygon1.min_prev_energy + polygon2.min_prev_energy
        aggregated_max_prev = polygon1.max_prev_energy + polygon2.max_prev_energy

        aggregated_polygon = DependencyPolygon(aggregated_min_prev, aggregated_max_prev, numsamples)

        if len(polygon1.points) == 2 and len(polygon2.points) == 2:
            # Special case: only two points (e.g., first timestep with min/max at 0)
            min_current_energy = polygon1.points[0].y + polygon2.points[0].y
            max_current_energy = polygon1.points[1].y + polygon2.points[1].y
            dependency_amount = polygon1.points[1].x + polygon2.points[1].x

            aggregated_polygon.add_point(dependency_amount, min_current_energy)
            aggregated_polygon.add_point(dependency_amount, max_current_energy)

        else:
            # General case: Iterate from min dependency to max dependency
            step1 = (polygon1.max_prev_energy - polygon1.min_prev_energy) / (numsamples - 1)
            step2 = (polygon2.max_prev_energy - polygon2.min_prev_energy) / (numsamples - 1)
            step = (aggregated_max_prev - aggregated_min_prev) / (numsamples - 1)

            for j in range(numsamples):
                current_prev_energy1 = polygon1.min_prev_energy + j * step1
                current_prev_energy2 = polygon2.min_prev_energy + j * step2
                current_prev_energy = aggregated_min_prev + j * step

                # Find or interpolate min/max energy usage for DFO1
                matching_points1 = find_or_interpolate_points(polygon1.points, current_prev_energy1)
                dfo1_min_energy = matching_points1[0].y
                dfo1_max_energy = matching_points1[1].y

                # Find or interpolate min/max energy usage for DFO2
                matching_points2 = find_or_interpolate_points(polygon2.points, current_prev_energy2)
                dfo2_min_energy = matching_points2[0].y
                dfo2_max_energy = matching_points2[1].y

                # Aggregate min/max energy
                min_current_energy = dfo1_min_energy + dfo2_min_energy
                max_current_energy = dfo1_max_energy + dfo2_max_energy

                aggregated_polygon.add_point(current_prev_energy, min_current_energy)
                aggregated_polygon.add_point(current_prev_energy, max_current_energy)

        aggregated_polygons.append(aggregated_polygon)

    aggregated_dfo = DFO(-1, [0], [0], numsamples)
    aggregated_dfo.polygons = aggregated_polygons
    return aggregated_dfo