from typing import List
from datetime import datetime, timedelta
from classes.DFO import Point, DependencyPolygon, DFO

def find_or_interpolate_points(points: List[Point], dependency_value: float) -> List[Point]:
    """Finds or interpolates points for a given dependency value."""
    matching_points = [point for point in points if point.x == dependency_value] # try find exact match

    if not matching_points:
        # Perform linear interpolation if an exact match is not found on points before and after dependency_value
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

def create_start_padding(num_padding: int, numsamples: int) -> List[DependencyPolygon]:
    """Generates start padding polygons with zero dependency and zero usage."""
    start_polygons = []
    
    for _ in range(num_padding):
        polygon = DependencyPolygon(0.0, 0.0, numsamples)
        polygon.points = [Point(0.0, 0.0), Point(0.0, 0.0)]  # Two (0,0) points
        start_polygons.append(polygon)

    return start_polygons


def create_end_padding(dfo: DFO, num_padding: int, numsamples: int) -> List[DependencyPolygon]:
    """Generates end padding polygons based on the last real polygon's total energy constraints."""
    if not dfo.polygons or num_padding <= 0:
        return []

    last_polygon = dfo.polygons[-1]

    min_total_energy = min(p.x + p.y for p in last_polygon.points)
    max_total_energy = max(p.x + p.y for p in last_polygon.points)

    end_polygons = []
    for _ in range(num_padding):
        polygon = DependencyPolygon(min_total_energy, max_total_energy, numsamples)
        polygon.points = [
            Point(min_total_energy, 0.0),
            Point(min_total_energy, 0.0),
            Point(max_total_energy, 0.0),
            Point(max_total_energy, 0.0)
        ]  # Four points (min_total_energy, 0) and (max_total_energy, 0)

        end_polygons.append(polygon)

    return end_polygons

def agg2to1(dfo1: DFO, dfo2: DFO, numsamples: int) -> DFO:
    """Aggregates two DFOs into one, handling misaligned start times by padding with temporary polygons."""
    
    # Determine the earliest start time
    start_time = min(dfo1.earliest_start, dfo2.earliest_start)

    # Compute how many padding polygons are needed at the start
    pad_start_1 = int((dfo1.earliest_start - start_time).total_seconds() // 3600)
    pad_start_2 = int((dfo2.earliest_start - start_time).total_seconds() // 3600)

    # Compute how many padding polygons are needed at the end
    max_length = max(len(dfo1.polygons) + pad_start_1, len(dfo2.polygons) + pad_start_2)
    pad_end_1 = max_length - (len(dfo1.polygons) + pad_start_1)
    pad_end_2 = max_length - (len(dfo2.polygons) + pad_start_2)

    # Create padded versions of the polygons
    padded_polygons_1 = (
        create_start_padding(pad_start_1, numsamples) +
        dfo1.polygons +
        create_end_padding(dfo1, pad_end_1, numsamples)
    )

    padded_polygons_2 = (
        create_start_padding(pad_start_2, numsamples) +
        dfo2.polygons +
        create_end_padding(dfo2, pad_end_2, numsamples)
    )

    # Aggregate the aligned polygons
    aggregated_polygons = []
    for i in range(max_length):
        polygon1 = padded_polygons_1[i]
        polygon2 = padded_polygons_2[i]

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

    # Create the aggregated DFO, preserving the aligned start time
    aggregated_dfo = DFO(-1, [0], [0], numsamples, earliest_start=start_time)
    aggregated_dfo.polygons = aggregated_polygons
    return aggregated_dfo

def aggnto1(dfos: List[DFO], numsamples: int) -> DFO:
    """Aggregates multiple DFOs into one using accumulating pairwise aggregation."""
    if not dfos:
        raise RuntimeError("No DFOs provided for aggregation. Kind Regards, aggnto1 function")

    # Start aggregation with the first DFO
    aggregated_dfo = dfos[0]

    # Aggregate subsequent DFOs
    for i in range(1, len(dfos)):
        aggregated_dfo = agg2to1(aggregated_dfo, dfos[i], numsamples)

    return aggregated_dfo
