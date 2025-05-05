import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from aggregation.clustering.metrics import evaluate_clustering
from config import config
from datetime import timedelta
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer, DFO, aggnto1
import matplotlib.pyplot as plt

CLUSTER_METHOD = config.CLUSTER_METHOD
CLUSTER_PARAMS = config.CLUSTER_PARAMS


def extract_features(offer):
    """
    Creates a feature vector for an offer:
      [earliest_ts, latest_ts, duration_secs, min_energy, total_energy]
    """
    if isinstance(offer, Flexoffer):
        return config.define_clustering_features_fo(offer)
    elif isinstance(offer, DFO):
        return config.define_clustering_features_dfo(offer)
    else:
        raise ValueError("Unknown offer type")


def cluster_offers(offers, n_clusters):
    """
    Clusters offers into up to n_clusters, normalizing features.

    Inputs:
        offers: List of FOs or DFOs
        n_clusters: Int

    output:
        list of clusters of length n_clusters and corresponding labels.
    """
    X = np.vstack([extract_features(o) for o in offers])
    X = StandardScaler().fit_transform(X)

    method = CLUSTER_METHOD.lower()

    params = dict(CLUSTER_PARAMS.get(method, {}))
    if method == 'ward' or method == 'kmeans':
        params['n_clusters'] = min(n_clusters, len(offers))
    elif method == 'gmm':
        params['n_components'] = min(n_clusters, len(offers))
    if method == 'ward':
        model = AgglomerativeClustering(params)
    elif method == 'kmeans':
        model = KMeans(params)
    elif method == 'gmm':
        model = GaussianMixture(params)
    elif method == 'dbscan':
        model = DBSCAN(params)
    else:
        raise ValueError(f"Unsupported CLUSTER_METHOD: {CLUSTER_METHOD}")

    labels = model.fit_predict(X)

    # Group offers by their cluster
    clustered_offers = {i: [] for i in range(n_clusters)}

    for offer, cluster_id in zip(offers, labels):
        clustered_offers[cluster_id].append(offer)

    return list(clustered_offers.values()), labels


def aggregate_clusters(clustered_offers):
    """
    Aggregates FlexOffers and DFOs per cluster
    """
    aggregated_offers = []

    for cluster in clustered_offers:
        flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
        dfos = [o for o in cluster if isinstance(o, DFO)]

        if flexoffers:
            afo = start_alignment_fast(flexoffers)
            aggregated_offers.append(afo)
        if dfos:
            afo = aggnto1(dfos, 4)
            aggregated_offers.append(afo)
    return aggregated_offers




def cluster_and_aggregate_flexoffers(offers, n_clusters=config.NUM_CLUSTERS):
    """
    Full pipeline: cluster, evaluate, and aggregate.
    """
    clustered_flexoffers, labels = cluster_offers(offers, n_clusters=n_clusters)

    # Compute clustering quality metrics
    evaluation = evaluate_clustering(offers, labels)

    print("\n===== Clustering Quality Metrics =====")
    print(f"Silhouette Score: {evaluation['Silhouette Score']:.3f}" if evaluation['Silhouette Score'] is not None else "Silhouette Score: N/A (only 1 cluster)")
    print(f"Davies-Bouldin Index: {evaluation['Davies-Bouldin Index']:.3f}" if evaluation['Davies-Bouldin Index'] is not None else "Davies-Bouldin Index: N/A (only 1 cluster)")
    print("======================================")

    return aggregate_clusters(clustered_flexoffers)


def aggregate_rolling_horizon(offers, now, window_h):
    """
    Returns aggregated offers whose start times are in [now, now+window_h).
    This can be used *before* scheduling.
    """
    window_end = now + timedelta(hours=window_h)
    active = [fo for fo in offers if fo.get_est() >= now and fo.get_est() < window_end]

    if not active:
        return []

    agg_offers, labels = cluster_and_aggregate_flexoffers(active)
    return agg_offers



# --- not used currently ---#



def meets_market_compliance(offer: Flexoffer) -> bool:
    if offer.get_min_overall_alloc() < config.MIN_BID_SIZE:
        return True
    return False


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
