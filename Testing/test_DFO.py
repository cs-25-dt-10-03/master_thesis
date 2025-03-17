import pytest
from datetime import datetime, timedelta
from flexoffer_logic import DFO, DependencyPolygon, Point, agg2to1, aggnto1, disagg1to2, disagg1toN, findOrInterpolatePoints
from classes.electricVehicle import ElectricVehicle
from optimization.DFOOptimizer import DFO_Optimization
from aggregation.clustering.Hierarchical_clustering import extract_features, cluster_offers, cluster_and_aggregate_flexoffers

@pytest.fixture
def charging_window_start():
    return datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)

@pytest.fixture
def duration():
    return timedelta(hours=3)

@pytest.fixture
def ev1():
    return ElectricVehicle(
        vehicle_id=1,
        capacity=75.0,
        soc_min=0.20,
        soc_max=0.80,
        charging_power=7.0,
        charging_efficiency=0.84,
    )

@pytest.fixture
def ev2():
    return ElectricVehicle(
        vehicle_id=2,
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
    )

@pytest.fixture
def ev3():
    return ElectricVehicle(
        vehicle_id=3,
        capacity=100.0,
        soc_min=0.70,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
    )


def test_create_dfos(charging_window_start, duration):
    """Tests creation of DFOs and aggregation using the C++ backend."""
    min_prev1 = [0, 5, 10]
    max_prev1 = [7, 12, 17]
    
    dfo1 = DFO(1, min_prev1, max_prev1, 4, 7, earliest_start=int(charging_window_start.timestamp()))
    dfo1.generate_dependency_polygons()

    min_prev2 = [0, 3, 8]
    max_prev2 = [5, 10, 15]
    
    dfo2 = DFO(2, min_prev2, max_prev2, 4, 7, earliest_start=int((charging_window_start + timedelta(hours=1)).timestamp()))
    dfo2.generate_dependency_polygons()

    dfo3 = agg2to1(dfo1, dfo2, 4)

    print(dfo3)
    assert isinstance(dfo3, DFO)
    assert len(dfo3.polygons) > 0


def test_dependency_polygon_generate_and_sort():
    """Tests that DependencyPolygon generates and sorts points correctly."""
    dp = DependencyPolygon(0, 10, 5)
    dp.generate_polygon(7, 2, 8)

    assert len(dp.points) > 0
    xs = [p.x for p in dp.points]
    assert xs == sorted(xs)


def test_DFO_generate_dependency_polygons():
    """Tests that DFO generates dependency polygons correctly."""
    min_prev = [10, 5, 0]
    max_prev = [15, 10, 5]

    dfo = DFO(1, min_prev, max_prev, 5, 7, 0, 15, int(datetime(2024, 1, 1, 8, 0).timestamp()))
    dfo.generate_dependency_polygons()
    print(dfo)

    assert isinstance(dfo, DFO)
    assert len(dfo.polygons) == len(min_prev) - 1


def test_agg2to1_and_aggnto1():
    """Tests aggregation functions agg2to1 and aggnto1 with the C++ backend."""
    t0 = int(datetime(2024, 1, 1, 8, 0).timestamp())

    min_prev1 = [5, 3]
    max_prev1 = [10, 8]
    dfo1 = DFO(1, min_prev1, max_prev1, 5, 7, earliest_start=t0)
    dfo1.generate_dependency_polygons()

    min_prev2 = [2, 1]
    max_prev2 = [7, 4]
    dfo2 = DFO(2, min_prev2, max_prev2, 5, 7, earliest_start=t0)
    dfo2.generate_dependency_polygons()

    aggregated_dfo = agg2to1(dfo1, dfo2, 5)
    assert isinstance(aggregated_dfo, DFO)
    assert len(aggregated_dfo.polygons) > 0

    aggregated_dfo_multi = aggnto1([dfo1, dfo2], 5)
    assert isinstance(aggregated_dfo_multi, DFO)
    assert len(aggregated_dfo_multi.polygons) > 0


def test_disagg1to2_and_disagg1toN(ev1, ev2, ev3, charging_window_start, duration):
    """Tests disaggregation functions disagg1to2 and disagg1toN using the C++ backend."""

    # Create DFOs from EVs, each with 3 time steps, but offset compared to one another
    dfo1 = ev1.create_dfo(charging_window_start, duration, numsamples=4)
    dfo2 = ev2.create_dfo(charging_window_start + timedelta(hours=1), duration, numsamples=4)
    dfo3 = ev3.create_dfo(charging_window_start + timedelta(hours=2), duration, numsamples=4)

    print(dfo1)
    print(dfo2)
    print(dfo3)

    dfos = [dfo1, dfo2, dfo3]
    # Aggregated DFO should have 5 time steps
    aggregated_dfo = aggnto1(dfos, numsamples=4)

    print(aggregated_dfo)

    # Ensure yA_ref matches the expected number of timesteps
    yA_ref = [4.0, 8.0, 10.0, 12.0, 6.0]

    # Disaggregate into multiple DFOs
    y_refs = disagg1toN(aggregated_dfo, dfos, yA_ref)
    print("Disaggregated y_refs:", y_refs)

    assert len(y_refs) == len(dfos)
    for i, dfo in enumerate(dfos):
        assert len(y_refs[i]) >= len(dfo.polygons)

    DFOs = cluster_and_aggregate_flexoffers(dfos, n_clusters=2)

    print(DFOs)

def test_DFO_Optimization(ev3, charging_window_start, duration):
    """Tests DFO optimization function with a simple cost structure."""
    cost_per_unit = [0.15, 0.20, 0.18]  # Example spot prices per timestep

    dfo1 = ev3.create_dfo(charging_window_start, duration, numsamples=4)

    # Ensure the cost array length matches the number of polygons
    assert len(cost_per_unit) == len(dfo1.polygons)

    # Run optimization
    optimized_schedule = DFO_Optimization(dfo1, cost_per_unit)

    # Assertions: Check that optimization produces valid results
    assert isinstance(optimized_schedule, list)
    assert len(optimized_schedule) == len(dfo1.polygons)

    # Ensure the optimized energy values are within the allowed range of the DFO
    dependency = 0.0
    for t, energy in enumerate(optimized_schedule):
        dependency += energy
        MinMaxPoints = findOrInterpolatePoints(dfo1.polygons[t].points, dependency)
        assert MinMaxPoints[0].y <= energy <= MinMaxPoints[1].y, \
            f"Energy {energy} at timestep {t} is out of bounds!"

    print("Optimized Schedule:", optimized_schedule)
    print(dfo1)