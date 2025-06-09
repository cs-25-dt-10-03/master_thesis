# capacitated.py

from math import ceil
from sklearn.cluster import AgglomerativeClustering
from flexoffer_logic import Flexoffer, DFO
import numpy as np

def _get_offer_energy(offer):
    """
    Returns the minimum guaranteed energy requirement for an offer:
    - For a Flexoffer, use its minimum overall allocation.
    - For a DFO, use its min_total_energy attribute.
    """
    if isinstance(offer, Flexoffer):
        return offer.get_min_overall_alloc()
    elif isinstance(offer, DFO):
        return offer.get_min_total_energy()

def enforce_cluster_energy_bounds(
    clusters: list,
    offers: list,
    X: np.ndarray,
    min_energy: float = None,
    max_energy: float = None,
    ward_params: dict = {}
) -> list:
    # 1) Split over‐cap clusters
    if max_energy is not None:
        clusters = []
        for c in clusters:
            total = sum(_get_offer_energy(o) for o in c)
            if total <= max_energy:
                clusters.append(c)
            else:
                # how many sub‐clusters we need
                k = ceil(total / max_energy)
                subX = X[[offers.index(o) for o in c]]
                ag = AgglomerativeClustering(n_clusters=k, **ward_params)
                labels = ag.fit_predict(subX)
                for lbl in set(labels):
                    clusters.append([c[i] for i,l in enumerate(labels) if l==lbl])

                    
    # 2) Merge under‐cap clusters
    if min_energy is not None:
        # build centroids & energies
        centroids = [np.mean(X[[offers.index(o) for o in c]], axis=0) for c in clusters]
        energies = [sum(_get_offer_energy(o) for o in c) for c in clusters]
        # greedy merge until all ≥ min_energy
        while True:
            idxs = [i for i,e in enumerate(energies) if e < min_energy]
            if not idxs:
                break
            i = idxs[0]
            # find closest neighbour
            dists = [np.linalg.norm(centroids[i]-centroids[j]) if j!=i else np.inf
                     for j in range(len(clusters))]
            j = int(np.argmin(dists))
            # merge j into i
            clusters[i].extend(clusters[j])
            # recompute centroid & energy
            idxs_i = [offers.index(o) for o in clusters[i]]
            centroids[i] = np.mean(X[idxs_i], axis=0)
            energies[i]   = sum(_get_offer_energy(o) for o in clusters[i])
            # remove j
            clusters.pop(j); centroids.pop(j); energies.pop(j)
    return clusters