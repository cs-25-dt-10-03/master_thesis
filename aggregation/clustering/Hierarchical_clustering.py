import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import List
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer
import matplotlib.pyplot as plt

def extract_features(flex_offers: List[Flexoffer]):
    features = []
    
    for fo in flex_offers:
        est = fo.get_est_hour()
        lst = fo.get_lst_hour()
        time_flexibility = lst - est
        total_energy = fo.get_total_energy()
        
        features.append([est, lst, time_flexibility, total_energy])
    
    return np.array(features)


def cluster_flexoffers(flex_offers: List[Flexoffer], n_clusters=3) -> dict[int, list]: #[cluster nummer, liste af fos]
    features = extract_features(flex_offers)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(features)

    clustered_flexoffers = {i: [] for i in range(n_clusters)}
    for fo, label in zip(flex_offers, labels):
        clustered_flexoffers[label].append(fo)

    return clustered_flexoffers


def aggregate_clusters(clustered_flexoffers):
    aggregated_flexoffers = []

    for cluster_id, flex_offer_group in clustered_flexoffers.items():
        if len(flex_offer_group) > 1:
            aggregated_flexoffer = start_alignment_fast(flex_offer_group)
            aggregated_flexoffers.append(aggregated_flexoffer)
        else:
            aggregated_flexoffers.append(flex_offer_group[0])
    return aggregated_flexoffers



def cluster_and_aggregate_flexoffers(flex_offers: List[Flexoffer], n_clusters=3):
    clustered_flexoffers = cluster_flexoffers(flex_offers, n_clusters=n_clusters)
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