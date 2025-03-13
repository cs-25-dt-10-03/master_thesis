import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from typing import List
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer, DFO, aggnto1
import matplotlib.pyplot as plt

def extract_features(offers: List[Flexoffer]):
    features = []
    
def extract_features(offer):
    """
    Extracts features for clustering from both FlexOffers and DFOs.
    """
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

def cluster_offers(offers, n_clusters=3):
    feature_vectors = np.array([extract_features(o) for o in offers])

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(feature_vectors)

    # Group offers by their cluster
    clustered_offers = {i: [] for i in range(n_clusters)}
    for offer, cluster_id in zip(offers, labels):
        clustered_offers[cluster_id].append(offer)

    return list(clustered_offers.values())


def aggregate_clusters(clustered_offers):
    aggregated_offers = []
    
    for cluster in clustered_offers:
        flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
        dfos = [o for o in cluster if isinstance(o, DFO)]

        if flexoffers:
            aggregated_offers.append(start_alignment_fast(flexoffers))
        if dfos:
            aggregated_offers.append(aggnto1(dfos, 5))

    return aggregated_offers


def cluster_and_aggregate_flexoffers(offers, n_clusters=3):
    clustered_flexoffers = cluster_offers(offers, n_clusters=n_clusters)
    return aggregate_clusters(clustered_flexoffers)



def visualize_clusters(flex_offers, labels):
    features = extract_features(flex_offers)
    
    plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.xlabel("Earliest Start Time (Unix)")
    plt.ylabel("Latest start time)")
    plt.title("Agglomerative Clustering of FlexOffers")
    plt.grid(True)
    plt.colorbar(label="Cluster ID")
    plt.show()


def plot_dendrogram(flex_offers, method="ward"):
    
    features = extract_features(flex_offers)

    linkage_matrix = linkage(features, method=method)

    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=[fo.get_offer_id() for fo in flex_offers], leaf_rotation=90)
    plt.xlabel("FlexOffer ID")
    plt.ylabel("Cluster Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()