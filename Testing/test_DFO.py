import pytest
from datetime import datetime, timedelta
from classes.flexOffer import FlexOffer
from classes.DFO import DFO
from classes.electricVehicle import ElectricVehicle
from aggregation.DFO_aggregation import agg2to1, aggnto1
import numpy as np
from classes.DFO import DFO, DependencyPolygon, Point

@pytest.fixture
def ev1():
    return ElectricVehicle(
        vehicle_id="TeslaModelY_1",
        capacity=75.0,
        soc_min=0.20,
        soc_max=0.80,
        charging_power=7.0,
        charging_efficiency=0.84,
    )

@pytest.fixture
def ev2():
    return ElectricVehicle(
        vehicle_id="TeslaModelS_1",
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
    )


def test_create_dfos(ev1, ev2):
    charging_window_start = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)
    charging_window_end = charging_window_start + timedelta(hours=8)
    duration = timedelta(hours=3) 

    dfo1 = ev1.create_dfo(charging_window_start,
                                   charging_window_end,
                                   duration,
                                   numsamples=4)

    dfo2 = ev2.create_dfo(charging_window_start+timedelta(hours=1),
                                   charging_window_end,
                                   duration,
                                   numsamples=4)
    dfo3 = agg2to1(dfo1, dfo2, 4)
    dfo3.plot_dfo()


def test_dependency_polygon_generate_and_sort():
    dp = DependencyPolygon(min_prev=0, max_prev=10, numsamples=5)
    dp.generate_polygon(charging_power=7, next_min_prev=2, next_max_prev=8)
    assert len(dp.points) > 0
    xs = [p.x for p in dp.points]
    assert xs == sorted(xs)

def test_DFO_generate_dependency_polygons():
    min_prev = [10, 5, 0]
    max_prev = [15, 10, 5]
    dfo = DFO(
        dfo_id=1,
        min_prev=min_prev,
        max_prev=max_prev,
        numsamples=5,
        charging_power=7,
        min_total_energy=0,
        max_total_energy=15,
        earliest_start=datetime(2024, 1, 1, 8, 0)
    )
    dfo.generate_dependency_polygons()
    assert len(dfo.polygons) == len(min_prev) - 1


def test_agg2to1_and_aggnto1():
    t0 = datetime(2024, 1, 1, 8, 0)
    min_prev1 = [5, 3]
    max_prev1 = [10, 8]
    dfo1 = DFO(
        dfo_id=1,
        min_prev=min_prev1,
        max_prev=max_prev1,
        numsamples=5,
        charging_power=7,
        earliest_start=t0
    )
    dfo1.polygons = [DependencyPolygon(5, 10, 5), DependencyPolygon(3, 8, 5)]
    for poly in dfo1.polygons:
        poly.points = [Point(5, 2), Point(10, 3)]
    
    min_prev2 = [2, 1]
    max_prev2 = [7, 4]
    dfo2 = DFO(
        dfo_id=2,
        min_prev=min_prev2,
        max_prev=max_prev2,
        numsamples=5,
        charging_power=7,
        earliest_start=t0
    )
    dfo2.polygons = [DependencyPolygon(2, 7, 5), DependencyPolygon(1, 4, 5)]
    for poly in dfo2.polygons:
        poly.points = [Point(2, 1), Point(7, 2)]
    
    aggregated_dfo = agg2to1(dfo1, dfo2, numsamples=5)
    assert isinstance(aggregated_dfo, DFO)
    assert len(aggregated_dfo.polygons) > 0

    aggregated_dfo_multi = aggnto1([dfo1, dfo2], numsamples=5)
    assert isinstance(aggregated_dfo_multi, DFO)
    assert len(aggregated_dfo_multi.polygons) > 0