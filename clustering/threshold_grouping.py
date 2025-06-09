# clustering/threshold_grouping.py

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import config
from clustering.grid_bucket import grid_bucket_clustering  # assumes you placed the helper there

def threshold_group_offers(offers):
    """
    Group FlexOffers by Chebyshev‐distance thresholds.

    Returns:
        clusters: List[List[Flexoffer]] — each sublist is one cluster
        labels:   np.ndarray of shape (N,) with cluster IDs
    """
    # 1) Build feature matrix for the chosen threshold set
    #    Here we assume 'time_range' ⇒ [est, lst]. Adapt if you support other sets.
    X = np.array([[fo.get_est(), fo.get_lst()] for fo in offers], dtype=float)

    # 2) Load per-dimension thresholds from config
    #    e.g. for 'time_range': [tau_est, tau_lst]
    thresholds = config.CLUSTER_THRESHOLDS[config.CLUSTER_THRESHOLD_SET]
    tau_list  = thresholds
    if len(tau_list) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} thresholds, got {len(tau_list)}")

    # 3) For Chebyshev distance, the scalar threshold is
    tau = max(tau_list)

    # 4) If grid‐bucket mode is enabled, do the O(N·D) binning
    buckets = defaultdict(list)
    for i, x in enumerate(X):
        # key j = floor(x_j / tau_j)
        key = tuple(int(math.floor(x[j] / tau_list[j])) for j in range(X.shape[1]))
        buckets[key].append(i)

    # build clusters and labels
    clusters_idx = list(buckets.values())
    clusters = [[offers[i] for i in idxs] for idxs in clusters_idx]

    labels = np.empty(len(offers), dtype=int)
    for label, idxs in enumerate(clusters_idx):
        for i in idxs:
            labels[i] = label

    config.NUM_CLUSTERS = len(clusters)
    return clusters, labels
