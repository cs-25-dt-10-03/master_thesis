from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from flexoffer_logic import Flexoffer, DFO
import numpy as np


def extract_features(offer):
    """
    Creates a feature vector for an offer:
      [earliest_ts, latest_ts, min_total_energy]
    """ 
    if isinstance(offer, Flexoffer):
        return np.array([
            offer.get_est(),
            offer.get_lst(),
            offer.get_duration(),
            offer.get_lst() - offer.get_est(),
            offer.get_min_overall_alloc(),
        ])
    elif isinstance(offer, DFO):
        return np.array([
            offer.get_est(),
            offer.get_lst(),
            offer.min_total_energy,
        ])
    else:
        raise ValueError("Unknown offer type")


def evaluate_clustering(offers, labels):
    features = np.array([extract_features(o) for o in offers])

    n_samples = len(labels)
    labels_set = set(labels)
    n_labels = len(labels_set)
    silhouette = None

    if 2 <= n_labels <= n_samples - 1:
        try:
            silhouette = silhouette_score(features, labels)
        except ValueError:
            silhouette = None

        davies_bouldin = davies_bouldin_score(features, labels)
    else:
        silhouette = None
        davies_bouldin = None

    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin
    }
