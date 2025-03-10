import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from aggregation.clustering.Hierarchical_clustering import extract_features, cluster_flexoffers, visualize_clusters, plot_dendrogram, aggregate_clusters
from aggregation.alignments import start_alignment_fast
from datetime import datetime, timedelta
from typing import List
from classes.electricVehicle import ElectricVehicle
import matplotlib.pyplot
from flexoffer_logic import Flexoffer, TimeSlice
from helpers import dt_to_unix

def simulate_mock_evs(offer_id, soc_min, soc_max, charging_power, capacity):
    ev = ElectricVehicle(
        vehicle_id=offer_id,
        capacity=capacity,
        soc_min=soc_min,
        soc_max=soc_max,
        charging_power=charging_power,
        charging_efficiency=0.95
    )
    return ev.create_flex_offer(tec_fo=True)

@pytest.fixture(params=[10, 100])
def fos(request):
    num_instances = request.param
    np.random.seed(42)

    fos = [
        simulate_mock_evs(
            offer_id=i,
            soc_min=np.random.uniform(3.1, 7.0),
            soc_max=np.random.uniform(0.6, 0.9),
            charging_power = np.random.choice([7.2, 11]),
            capacity=np.random.randint(70, 100)
        )
        for i in range(num_instances)
    ]

    return fos

def test_cluster_and_aggregate_flexoffers(fos, n_clusters=3):
    clustered_flexoffers = cluster_flexoffers(fos, n_clusters=n_clusters)
    aggregated_offers = aggregate_clusters(clustered_flexoffers)


    print("\n=== Aggregated FlexOffers ===\n")
    for i, afo in enumerate(aggregated_offers):
        print(f"Aggregated FlexOffer {i+1}:")
        afo.print_flexoffer()
        print("\n" + "="*50 + "\n")
