import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from aggregation.clustering.metrics import evaluate_clustering
from typing import List
from config import config
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer, DFO, aggnto1
import matplotlib.pyplot as plt

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

def cluster_offers(offers, n_clusters=3):
    feature_vectors = np.array([extract_features(o) for o in offers])

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(feature_vectors)

    # Group offers by their cluster
    clustered_offers = {i: [] for i in range(n_clusters)}
    for offer, cluster_id in zip(offers, labels):
        clustered_offers[cluster_id].append(offer)

    return list(clustered_offers.values()), labels


def aggregate_clusters(clustered_offers):
    aggregated_offers = []
    
    for cluster in clustered_offers:
        flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
        dfos = [o for o in cluster if isinstance(o, DFO)]

        if flexoffers:
            afo = start_alignment_fast(flexoffers)
            if meets_market_compliance(afo):
                aggregated_offers.append(afo)
            else:
                aggregated_offers.append(afo)
                print(f"⚠️ Warning: Cluster did not meet market compliance. Adjusting...")
        if dfos:
            afo = aggnto1(dfos, 4)
            aggregated_offers.append(afo)
    return aggregated_offers

def meets_market_compliance(offer: Flexoffer) -> bool:
    if offer.get_min_overall_alloc() < config.MIN_BID_SIZE:
        return True
    return False

def cluster_and_aggregate_flexoffers(offers, n_clusters=3):
    clustered_flexoffers, labels = cluster_offers(offers, n_clusters=n_clusters)

    # Compute clustering quality metrics
    evaluation = evaluate_clustering(offers, labels)
    
    print("\n===== Clustering Quality Metrics =====")
    print(f"Silhouette Score: {evaluation['Silhouette Score']:.3f}" if evaluation['Silhouette Score'] is not None else "Silhouette Score: N/A (only 1 cluster)")
    print(f"Davies-Bouldin Index: {evaluation['Davies-Bouldin Index']:.3f}" if evaluation['Davies-Bouldin Index'] is not None else "Davies-Bouldin Index: N/A (only 1 cluster)")
    print("======================================")

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


def needed_for_compliance(offer):
    if offer.get_total_energy() < config.MIN_BID_SIZE:
        additional_energy_needed = config.MIN_BID_SIZE - offer.get_min_overall_alloc()
        print(additional_energy_needed)
    if config.REQUIRE_UNIFORM:
        avg_energy = sum(offer.get_scheduled_allocation()) / offer.get_duration()
        print(avg_energy)
    return offer