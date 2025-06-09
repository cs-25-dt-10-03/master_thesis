import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from config import config
from clustering.distance_factory import get_distance_metric
from clustering.threshold_grouping import threshold_group_offers

def cluster_on_features(X: np.ndarray, offers: list, n_clusters: int):
    """
    Pure clustering on feature‐matrix X.
    Respects AGGLO_USE_THRESHOLD for 'ward', else fixed-k.
    """
    if config.USE_THRESHOLD:
        return threshold_group_offers(offers)
    
    method = config.CLUSTER_METHOD.lower()
    params = config.CLUSTER_PARAMS.get(method, {}).copy()

    if method == 'ward':
        params['n_clusters'] = min(n_clusters, len(offers))
        model = AgglomerativeClustering(
            metric=get_distance_metric(),
            linkage=config.CLUSTER_LINKAGE,
            **params
        )

    elif method == 'kmeans':
        params['n_clusters'] = min(n_clusters, len(offers))
        model = KMeans(**params)

    elif method == 'gmm':
        params['n_components'] = min(n_clusters, len(offers))
        model = GaussianMixture(**params)

    elif method == 'dbscan':
        model = DBSCAN(**params)

    else:
        raise ValueError(f"Unsupported CLUSTER_METHOD: {config.CLUSTER_METHOD}")

    labels = model.fit_predict(X)
    # build clusters as lists of offer objects
    clusters = []
    for lbl in sorted(set(labels)):
        clusters.append([o for o, l in zip(offers, labels) if l == lbl])
    return clusters, labels
