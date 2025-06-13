# clustering/threshold_grouping.py

import math
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from config import config

import math
from collections import defaultdict
import numpy as np

def threshold_group_offers(offers):
    """
    Group FlexOffers by Chebyshev‐distance thresholds.

    Returns:
        clusters: List[List[Flexoffer]] — each sublist is one cluster
        labels:   np.ndarray of shape (N,) with cluster IDs
    """
    # 1) Build feature matrix for the chosen threshold set
    X = np.array([[fo.get_est(), fo.get_lst(), fo.get_et()] for fo in offers], dtype=float)

    # 2) Load per-dimension thresholds from config
    thresholds = config.CLUSTER_THRESHOLDS[config.CLUSTER_THRESHOLD_SET]

    # 4) split into cells
    cells = defaultdict(list)

    # (3,2)
    for i, x in enumerate(X):
        key  = tuple(int(math.floor(x[j] / thresholds[j])) for j in range(X.shape[1]))
        cells[key].append(i)

    # build clusters and labels
    clusters_idx = list(cells.values())
    clusters = [[offers[i] for i in idxs] for idxs in clusters_idx]

    labels = np.empty(len(offers), dtype=int)
    for label, idxs in enumerate(clusters_idx):
        for i in idxs:
            labels[i] = label

    config.NUM_CLUSTERS = len(clusters)
    return clusters, labels
