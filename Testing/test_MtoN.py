import numpy as np
import pytest
import datetime
from optimization.flexOfferOptimizer import optimize
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from classes.electricVehicle import ElectricVehicle
from database.parser import fo_parser
from config import config

CONFIG = config()


def simulate_mock_evs(offer_id, soc_min, soc_max, charging_power, capacity):
    ev = ElectricVehicle(
        vehicle_id=offer_id,
        capacity=capacity,
        soc_min=soc_min,
        soc_max=soc_max,
        charging_power=charging_power,
        charging_efficiency=0.95
    )
    return ev.create_synthetic_flex_offer(tec_fo=True)


@pytest.fixture(params=[10])
def fos(request):
    num_instances = request.param
    np.random.seed(42)

    fos = [
        simulate_mock_evs(
            offer_id=i,
            soc_min=np.random.uniform(0.3, 0.4),
            soc_max=np.random.uniform(0.6, 0.9),
            charging_power=np.random.choice([7.2, 11]),
            capacity=np.random.randint(70, 100)
        )
        for i in range(num_instances)
    ]

    return fos


def test_cluster_and_aggregate_flexoffers_mock_data(fos, n_clusters=3):
    aggregated_offers = cluster_and_aggregate_flexoffers(fos, n_clusters=3)

    print("\n=== Aggregated FlexOffers ===\n")
    for i, afo in enumerate(aggregated_offers):
        optimize(afo)
        print(f"Aggregated FlexOffer {i+1}:")
        afo.print_flexoffer()
        print("\n" + "="*50 + "\n")


def test_cluster_and_aggregate_flexoffers():
    dt1 = (datetime.datetime(2023, 12, 31, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    dt2 = (datetime.datetime(2024, 12, 31, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    fos = fo_parser(dt1, dt2)
    print("Len of fos: ", len(fos))
    afo = cluster_and_aggregate_flexoffers(fos, 100)

    optimize(afo)
    print("\n=== Aggregated FlexOffers ===\n")
    for i, fo in enumerate(afo):
        print(f"Aggregated FlexOffer {i+1}:")
        fo.print_flexoffer()
