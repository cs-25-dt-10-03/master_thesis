import numpy as np
import pytest
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from datetime import datetime, timedelta
from flexoffer_logic import Flexoffer, TimeSlice


def create_mock_flexoffer(offer_id, est_offset, lst_offset, duration, min_power, max_power):
    now = datetime.now()
    est = int((now + timedelta(hours=est_offset)).timestamp())
    lst = int((now + timedelta(hours=lst_offset)).timestamp())  # Convert to Unix time
    end = int((now + timedelta(hours=lst_offset + duration)).timestamp())

    profile = [TimeSlice(min_power, max_power) for _ in range(duration)]
    return Flexoffer(offer_id, est, lst, end, profile, duration)


@pytest.fixture(params=[10, 100, 1000])
def fos(request):
    num_instances = request.param
    np.random.seed(42)

    flex_offers = [
        create_mock_flexoffer(
            offer_id=i,
            est_offset=np.random.randint(0, 24),
            lst_offset=np.random.randint(0, 24),
            duration=np.random.randint(1, 5),
            min_power=np.random.uniform(0.5, 3.0),
            max_power=np.random.uniform(3.1, 7.0),
        )
        for i in range(num_instances)
    ]

    return flex_offers


def test_clustering(fos):
    clustered_fos = cluster_and_aggregate_flexoffers(fos, n_clusters=3)
