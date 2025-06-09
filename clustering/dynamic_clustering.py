import numpy as np
from math import inf
from config import config
from clustering.clustering_core import cluster_on_features
from clustering.metrics import evaluate_clustering, compute_flex_loss_score

def dynamic_k_search(X: np.ndarray, offers: list, max_k: int = None):
    """
    Searches K in [CLUSTER_K_MIN, CLUSTER_K_MAX] (optionally capped by max_k),
    optimizing either silhouette or flex-loss.
    """
    k_min = config.CLUSTER_K_MIN
    k_max = min(config.CLUSTER_K_MAX, len(offers) - 1)
    if max_k is not None:
        k_max = min(k_max, max_k)

    best_k = k_min
    best_score = -inf if config.CLUSTER_SELECTION_METRIC == 'silhouette' else inf

    for k in range(k_min, k_max + 1):
        clusters, labels = cluster_on_features(X, offers, k)
        if config.CLUSTER_SELECTION_METRIC == 'silhouette':
            score = evaluate_clustering(offers, labels)["Silhouette Score"] or -inf
            improve = (score > best_score)
        else:
            score = compute_flex_loss_score(offers, labels)
            improve = (score < best_score)

        if improve or k == k_min:
            best_score, best_k = score, k

    return cluster_on_features(X, offers, best_k)
