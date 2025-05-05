import pytest
from datetime import datetime, timedelta
from flexoffer_logic import DFO, DependencyPolygon, Point, agg2to1, aggnto1, disagg1to2, disagg1toN, findOrInterpolatePoints
from classes.electricVehicle import ElectricVehicle
from optimization.DFOOptimizer import DFO_Optimization, DFO_MultiMarketOptimization, optimize_dfos
from aggregation.clustering.Hierarchical_clustering import extract_features, cluster_offers, cluster_and_aggregate_flexoffers
import pandas as pd

@pytest.fixture
def charging_window_start():
    return datetime.now().replace(year=2024, hour=22, minute=0, second=0, microsecond=0)

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
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
    )


def test_create_dfos(charging_window_start, duration):
    """Tests creation of DFOs and aggregation using the C++ backend."""
    min_prev1 = [0, 5, 10]
    max_prev1 = [7, 12, 17]
    
    dfo1 = DFO(1, min_prev1, max_prev1, 4, 7, earliest_start_time=int(charging_window_start.timestamp()))
    dfo1.generate_dependency_polygons()

    min_prev2 = [0, 3, 8]
    max_prev2 = [5, 10, 15]
    
    dfo2 = DFO(2, min_prev2, max_prev2, 4, 7, earliest_start_time=int((charging_window_start + timedelta(hours=1)).timestamp()))
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
    dfo1 = DFO(1, min_prev1, max_prev1, 5, 7, earliest_start_time=t0)
    dfo1.generate_dependency_polygons()

    min_prev2 = [2, 1]
    max_prev2 = [7, 4]
    dfo2 = DFO(2, min_prev2, max_prev2, 5, 7, earliest_start_time=t0)
    dfo2.generate_dependency_polygons()

    aggregated_dfo = agg2to1(dfo1, dfo2, 5)


    print("max energy for debugging")
    for poly in dfo1.polygons:
        print(poly.min_prev_energy)

    for poly in dfo2.polygons:
        print(poly.min_prev_energy)

    for poly in aggregated_dfo.polygons:
        print(poly.min_prev_energy)

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
'''
def test_DFO_Optimization(ev3, charging_window_start, duration):
    """Tests DFO optimization function with a simple cost structure."""

    dfo1 = ev3.create_dfo(charging_window_start, duration, numsamples=4)

    # Run optimization
    optimized_schedule = DFO_Optimization(dfo1)

    # Assertions: Check that optimization produces valid results
    assert isinstance(optimized_schedule, list)
    assert len(optimized_schedule) == len(dfo1.polygons)

    # Ensure the optimized energy values are within the allowed range of the DFO
    print("Optimized Schedule:", optimized_schedule)
    print(dfo1)
    dependency = 0.0
    for t, energy in enumerate(optimized_schedule):
        print("POints ", dfo1.polygons[t].points)
        MinMaxPoints = findOrInterpolatePoints(dfo1.polygons[t].points, dependency)
        dependency += energy
        print("MinMaxPoints: ", MinMaxPoints)
        print("Iteration: ", t)
        tolerance = 1e-6  # Define a small tolerance
        assert MinMaxPoints[0].y - tolerance <= energy <= MinMaxPoints[1].y + tolerance, \
            f"Energy {energy} at timestep {t} is out of bounds!"

'''
def test_DFO_MultiMarketOptimization(ev3, charging_window_start, duration):
    """🧠Tests mFRR-based multi-market optimization for a single DFO.🧠"""

    dfo = ev3.create_dfo(charging_window_start, duration, numsamples=4)
    print(dfo.get_est(), dfo.get_et())
    num_timesteps = len(dfo.polygons)

    # Run the optimization
    results_df = DFO_MultiMarketOptimization(dfo)

    # Basic sanity checks
    assert not results_df.empty
    assert "charge_kW" in results_df.columns
    assert len(results_df["charge_kW"]) >= num_timesteps

    # Validate energy allocation respects dependency constraints
    cumulative_energy = 0.0
    for t in range(num_timesteps):
        energy = results_df.loc[t, "charge_kW"]
        polygon = dfo.polygons[t]
        minmax = findOrInterpolatePoints(polygon.points, cumulative_energy)
        assert minmax[0].y <= energy <= minmax[1].y, \
            f"Energy {energy:.2f} at timestep {t} not within allowed bounds [{minmax[0].y:.2f}, {minmax[1].y:.2f}]"
        cumulative_energy += energy

    print("Multi-market optimization test passed.")
    print(results_df)


def test_optimize_dfos(ev1, ev2, ev3, charging_window_start, duration):
    """Tests the full multi-DFO optimization pipeline including joint timeline padding and multi-market solving."""

    # Create a list of DFOs from different EVs
    dfo1 = ev1.create_dfo(charging_window_start, duration, numsamples=4)
    dfo2 = ev2.create_dfo(charging_window_start + timedelta(hours=1), duration, numsamples=4)
    dfo3 = ev3.create_dfo(charging_window_start + timedelta(hours=2), duration, numsamples=4)
    dfos = [dfo1, dfo2, dfo3]

    # Run full optimization
    sol = optimize_dfos(dfos)
    print("Optimization Solution:", sol)

    # Check presence of key result fields
    assert "p" in sol
    assert isinstance(sol["p"], dict)

    # Check that each DFO index has an optimized schedule
    for i in range(len(dfos)):
        assert i in sol["p"], f"Missing solution for DFO {i}"
        alloc = sol["p"][i]
        assert isinstance(alloc, dict), f"Allocation for DFO {i} is not a dict"
        assert all(isinstance(val, (float, int)) or val is None for val in alloc.values()), f"Non-numeric values in allocation for DFO {i}"
        assert any(val is not None and not pd.isna(val) and val >= 0.0 for val in alloc.values()), f"DFO {i} has no energy allocated."

        # Print for visual confirmation
        print(f"\nDFO {i} Allocation: {alloc}")

    print("✅ Full DFO optimization test passed.")
