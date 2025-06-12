import numpy as np
from time import time
from sklearn.preprocessing import StandardScaler

from config import config
from clustering.feature_factory import extract_features
from clustering.clustering_core import cluster_on_features
from clustering.dynamic_clustering import dynamic_k_search
from clustering.capacitated import enforce_cluster_energy_bounds
from aggregation.aggregation import aggregate_clusters  # your original aggregation.py


def cluster_and_aggregate_offers(offers: list):
    """
    1) Extract & scale features
    2) Static vs dynamic k
    3) Optional energy‐cap splitting
    4) Aggregate
    """
    if not offers:
        return [], 0.0, 0.0

    # 1) features + scale
    feats = np.vstack([extract_features(o) for o in offers]).astype(np.float32)
    X = StandardScaler().fit_transform(feats)

    K = config.NUM_CLUSTERS

    # 3) run clustering
    t0 = time()
    if config.DYNAMIC_CLUSTERING:
        clusters, labels = dynamic_k_search(X, offers, max_k=K)
    else:
        clusters, labels = cluster_on_features(X, offers, n_clusters=K)
    t_cluster = time() - t0

    if config.MIN_CLUSTER_ENERGY is not None or config.MAX_CLUSTER_ENERGY is not None:
        ward_params = config.CLUSTER_PARAMS['ward'].copy()
        ward_params.pop('n_clusters', None)
        clusters = enforce_cluster_energy_bounds(
            clusters, offers, X,
            min_energy=config.MIN_CLUSTER_ENERGY,
            max_energy=config.MAX_CLUSTER_ENERGY,
            ward_params=ward_params
        )
        
    # 5) aggregate
    agg_offers, t_agg = aggregate_clusters(clusters)

    # Check that energy of FO == AFO (just a quick check to catch potential errors)
    if config.TYPE == "FO":
        total_individual = sum(fo.get_min_overall_alloc() for fo in offers)
        total_aggregated = sum(afo.get_min_overall_alloc() for afo in agg_offers)
        if abs(total_individual - total_aggregated) > 1e-6:
            raise ValueError(
                f"Clustering energy mismatch: "
                f"sum(individual)={total_individual:.3f}, "
                f"sum(aggregated)={total_aggregated:.3f}"
            )

    return agg_offers, t_cluster, t_agg
