# clustering/grid_bucket.py
import math
from collections import defaultdict
import numpy as np

def grid_bucket_clustering(X: np.ndarray, tau: float):
    """
    Cluster rows of X under Chebyshev distance by binning into cubes of side tau.
    Returns a list of clusters, each as a list of row-indices.
    """
    buckets = defaultdict(list)
    for i, x in enumerate(X):
        key = tuple(int(math.floor(coord / tau)) for coord in x)
        buckets[key].append(i)
    return list(buckets.values())
