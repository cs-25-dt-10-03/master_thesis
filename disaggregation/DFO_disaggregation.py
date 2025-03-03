from typing import List
from aggregation.DFO_aggregation import find_or_interpolate_points
from classes.DFO import DFO 

def disagg1to2(D1: DFO, D2: DFO, DA: DFO, yA_ref: List[float]) -> tuple[List[float], List[float]]:
    """
    Disaggregates a single aggregated DFO (DA) into two original DFOs (D1 and D2), accounting for start time offsets.
    
    Args:
        D1 (DFO): First original DFO.
        D2 (DFO): Second original DFO.
        DA (DFO): Aggregated DFO.
        yA_ref (List[float]): Reference schedule for the aggregated DFO.
    
    Returns:
        y1_ref (List[float]): Disaggregated schedule for DFO 1.
        y2_ref (List[float]): Disaggregated schedule for DFO 2.
    """
    T = len(DA.polygons)  # Number of timesteps
    if T != len(yA_ref):
        raise RuntimeError("Mismatch between DA timesteps and yA_ref size. Kind regards, disagg1to2 function")

    # Determine offsets in start time
    start_time = min(D1.earliest_start, D2.earliest_start)
    offset_1 = int((D1.earliest_start - start_time).total_seconds() // 3600)
    offset_2 = int((D2.earliest_start - start_time).total_seconds() // 3600)

    # Determine how many timesteps each DFO overlaps with DA
    overlap_start = max(offset_1, offset_2)
    overlap_end = min(len(D1.polygons) + offset_1, len(D2.polygons) + offset_2, T)

    # Initialize output lists
    y1_ref = [0.0] * len(D1.polygons)
    y2_ref = [0.0] * len(D2.polygons)

    dA, d1, d2 = 0.0, 0.0, 0.0

    # Assign reference values directly for non-overlapping start
    for i in range(offset_2):
        y1_ref[i] = yA_ref[i]  # Assign entire aggregated schedule to DFO1 if it starts earlier
        d1 += yA_ref[i]
        dA += yA_ref[i]
    for i in range(offset_1):
        y2_ref[i] = yA_ref[i]  # Assign entire aggregated schedule to DFO2 if it starts earlier
        d2 += yA_ref[i]
        dA += yA_ref[i]

    # Handle overlapping section using normal interpolation logic
    for i in range(overlap_start, overlap_end):
        polygonA = DA.polygons[i]
        polygon1 = D1.polygons[i - offset_1]
        polygon2 = D2.polygons[i - offset_2]

        # Find points with the respective energy dependency for DFO A, DFO 1, and DFO 2
        matching_pointsA = find_or_interpolate_points(polygonA.points, dA)
        matching_points1 = find_or_interpolate_points(polygon1.points, d1)
        matching_points2 = find_or_interpolate_points(polygon2.points, d2)

        # Calculate scaling factor f based on the reference schedule
        pointA_1, pointA_2 = matching_pointsA[0], matching_pointsA[1]
        f = 0 if (pointA_2.y - pointA_1.y) == 0 else (yA_ref[i] - pointA_1.y) / (pointA_2.y - pointA_1.y)

        # Use scaling factor on DFO 1 and 2 to determine their energy usage
        point1_1, point1_2 = matching_points1[0], matching_points1[1]
        y1_ref[i - offset_1] = point1_1.y + f * (point1_2.y - point1_1.y)

        point2_1, point2_2 = matching_points2[0], matching_points2[1]
        y2_ref[i - offset_2] = point2_1.y + f * (point2_2.y - point2_1.y)

        # Update dependency amounts
        dA += yA_ref[i]
        d1 += y1_ref[i - offset_1]
        d2 += y2_ref[i - offset_2]

    # Assign reference values directly for non-overlapping end
    for i in range(overlap_end, T):
        if i >= len(D1.polygons) + offset_1:
            y2_ref[i - offset_2] = yA_ref[i]  # Assign remaining schedule to DFO2 if it extends longer
        if i >= len(D2.polygons) + offset_2:
            y1_ref[i - offset_1] = yA_ref[i]  # Assign remaining schedule to DFO1 if it extends longer

    return y1_ref, y2_ref


def disagg1toN(DA: DFO, DFOs: List[DFO], yA_ref: List[float]) -> List[List[float]]:
    """
    Disaggregates a single aggregated DFO (DA) into multiple original DFOs (DFOs).
    
    Args:
        DA (DFO): Aggregated DFO.
        DFOs (List[DFO]): List of original DFOs.
        yA_ref (List[float]): Reference schedule for the aggregated DFO.
    
    Returns:
        y_refs (List[List[float]]): Disaggregated schedules for each DFO.
    """
    T = len(DA.polygons)  # Number of timesteps
    N = len(DFOs)         # Number of DFOs

    if T != len(yA_ref):
        raise RuntimeError("Mismatch between DA timesteps and yA_ref size. Kind regards, disagg1toN function")

    # Initialize energy dependency amounts for all DFOs
    d = [0.0] * N  # Dependency amounts for individual DFOs
    dA = 0.0       # Dependency amount for aggregated DFO

    # Initialize output list
    y_refs = [[0.0] * T for _ in range(N)]

    for i in range(T):
        # Get the polygon slice for the timestep
        polygonA = DA.polygons[i]

        # Find points with the respective energy dependency for aggregated DFO A
        matching_pointsA = find_or_interpolate_points(polygonA.points, dA)

        # Calculate scaling factor f based on the reference schedule
        pointA1, pointA2 = matching_pointsA[0], matching_pointsA[1]
        f = 0 if (pointA2.y - pointA1.y) == 0 else (yA_ref[i] - pointA1.y) / (pointA2.y - pointA1.y)

        # Disaggregate for each individual DFO
        for j in range(N):
            polygon = DFOs[j].polygons[i]
            matching_points = find_or_interpolate_points(polygon.points, d[j])

            point1, point2 = matching_points[0], matching_points[1]
            y_refs[j][i] = point1.y + f * (point2.y - point1.y)

            # Update dependency amount for the current DFO
            d[j] += y_refs[j][i]

        # Update dependency amount for the aggregated DFO
        dA += yA_ref[i]

    return y_refs
