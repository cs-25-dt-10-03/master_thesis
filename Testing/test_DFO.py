import pytest
from datetime import datetime, timedelta
from aggregation.DFO_aggregation import agg2to1, aggnto1
from classes.electricVehicle import ElectricVehicle
import numpy as np
from flexoffer_logic import DFO, DependencyPolygon, Point

@pytest.fixture
def charging_window_start():
    return datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)

@pytest.fixture
def charging_window_end(charging_window_start):
    return charging_window_start + timedelta(hours=8)

@pytest.fixture
def duration():
    return timedelta(hours=3)


def test_create_dfos(charging_window_start, charging_window_end, duration):
    """Tests creation of DFOs and aggregation."""
    min_prev1 = [0, 5, 10]
    max_prev1 = [7, 12, 17]
    
    dfo1 = DFO(
        1, min_prev1, max_prev1, 4, 7, earliest_start=int(charging_window_start.timestamp())
    )
    dfo1.generate_dependency_polygons()

    min_prev2 = [0, 3, 8]
    max_prev2 = [5, 10, 15]
    
    dfo2 = DFO(
        2, min_prev2, max_prev2, 4, 7, earliest_start=int((charging_window_start + timedelta(hours=1)).timestamp())
    )
    dfo2.generate_dependency_polygons()

    dfo3 = agg2to1(dfo1, dfo2, 4)

    print(dfo3)
    assert isinstance(dfo3, DFO)
    assert len(dfo3.polygons) > 0


def test_dependency_polygon_generate_and_sort():
    """Tests that DependencyPolygon generates and sorts points correctly."""
    dp = DependencyPolygon(0, 10, 5)  # Fixed: Use positional arguments
    dp.generate_polygon(7, 2, 8)
    
    assert len(dp.points) > 0
    xs = [p.x for p in dp.points]
    assert xs == sorted(xs)


def test_DFO_generate_dependency_polygons():
    """Tests that DFO generates dependency polygons correctly."""
    min_prev = [10, 5, 0]
    max_prev = [15, 10, 5]

    dfo = DFO(
        1, min_prev, max_prev, 5, 7, 0, 15, int(datetime(2024, 1, 1, 8, 0).timestamp())
    )

    dfo.generate_dependency_polygons()
    print(dfo)

    assert isinstance(dfo, DFO)
    assert len(dfo.polygons) == len(min_prev) - 1


def test_agg2to1_and_aggnto1():
    """Tests aggregation functions agg2to1 and aggnto1."""
    t0 = int(datetime(2024, 1, 1, 8, 0).timestamp())  # Fixed: Convert timestamp to int

    min_prev1 = [5, 3]
    max_prev1 = [10, 8]

    dfo1 = DFO(1, min_prev1, max_prev1, 5, 7, earliest_start=t0)
    dfo1.polygons = [DependencyPolygon(5, 10, 5), DependencyPolygon(3, 8, 5)]
    for poly in dfo1.polygons:
        poly.points = [Point(5, 2), Point(10, 3)]
    
    min_prev2 = [2, 1]
    max_prev2 = [7, 4]

    dfo2 = DFO(2, min_prev2, max_prev2, 5, 7, earliest_start=t0)
    dfo2.polygons = [DependencyPolygon(2, 7, 5), DependencyPolygon(1, 4, 5)]
    for poly in dfo2.polygons:
        poly.points = [Point(2, 1), Point(7, 2)]
    
    aggregated_dfo = agg2to1(dfo1, dfo2, 5)
    assert isinstance(aggregated_dfo, DFO)
    assert len(aggregated_dfo.polygons) > 0

    aggregated_dfo_multi = aggnto1([dfo1, dfo2], 5)
    assert isinstance(aggregated_dfo_multi, DFO)
    assert len(aggregated_dfo_multi.polygons) > 0
