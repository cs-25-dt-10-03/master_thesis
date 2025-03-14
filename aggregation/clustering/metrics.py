from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from flexoffer_logic import Flexoffer, DFO
import numpy as np

def extract_features(offer):
    if isinstance(offer, Flexoffer):
        return np.array([
            offer.get_est_hour(),
            offer.get_lst_hour(),
            offer.get_min_overall_alloc(),
        ])
    elif isinstance(offer, DFO):
        return np.array([
            offer.get_est_hour(),
            offer.get_lst_hour(),
            offer.min_total_energy,
        ])
    else:
        raise ValueError("Unknown offer type")

def evaluate_clustering(offers, labels):
    features = np.array([extract_features(o) for o in offers])

    if len(set(labels)) > 1:
        silhouette = silhouette_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
    else:
        silhouette = None
        davies_bouldin = None

    return {
        "Silhouette Score": silhouette,
        "Davies-Bouldin Index": davies_bouldin
    }
