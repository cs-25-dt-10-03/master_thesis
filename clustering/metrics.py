from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import euclidean
from flexoffer_logic import Flexoffer, DFO
from collections import defaultdict
import numpy as np
from clustering.feature_factory import extract_features

def evaluate_clustering(offers: list, labels: np.ndarray):
    feats = np.vstack([extract_features(o) for o in offers])
    n = len(labels)
    unique = set(labels)
    sil = db = None

    if 2 <= len(unique) <= n - 1:
        sil = silhouette_score(feats, labels, sample_size=min(1000, n), random_state=42)
        db  = davies_bouldin_score(feats, labels)

    return {"Silhouette Score": sil, "Davies-Bouldin Index": db}

def compute_flex_loss_score(offers: list, labels: np.ndarray):
    groups = defaultdict(list)
    for i, lbl in enumerate(labels):
        groups[lbl].append(i)

    total_loss = 0.0
    for members in groups.values():
        if not members:
            continue
        est = [offers[i].get_est() for i in members]
        lst = [offers[i].get_lst() for i in members]
        sum_spans = sum(lst[i] - est[i] for i in range(len(members)))
        if sum_spans > 0:
            span = max(lst) - min(est)
            total_loss += 1 - (span / sum_spans)
    return total_loss / len(groups) if groups else 0.0
