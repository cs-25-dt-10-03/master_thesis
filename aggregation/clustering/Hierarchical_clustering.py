import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import os
import multiprocessing
from time import time
from sklearn.preprocessing import StandardScaler
from config import config
from datetime import timedelta
from math import inf
from joblib import Parallel, delayed
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer, DFO, aggnto1, balance_alignment_aggregate, balance_alignment_tree_merge
import matplotlib.pyplot as plt
from aggregation.clustering.metrics import evaluate_clustering

# Note: Use config.CLUSTER_METHOD dynamically to allow runtime changes
# Remove static CLUSTER_METHOD and CLUSTER_PARAMS definitions

MAX_JOBS = min(multiprocessing.cpu_count(), config.PARALLEL_N_JOBS)

def extract_features(offer):
    """
    Creates a feature vector for an offer:
      [earliest_ts, latest_ts, min_total_energy]
    """
    if isinstance(offer, Flexoffer):
        feats = config.define_clustering_features_fo(offer)
        return feats
    elif isinstance(offer, DFO):
        feats = config.define_clustering_features_dfo(offer)
        return feats
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
    print(f"[Clustering] Extracting features for {len(offers)} offers...")

    X = np.vstack([extract_features(o) for o in offers])
    print("[Clustering] Scaling features...")
    X = StandardScaler().fit_transform(X)

    # Use dynamic config value for clustering method
    method = config.CLUSTER_METHOD.lower()
    params = config.CLUSTER_PARAMS.get(method, {})

    # Adjust parameters based on method
    if method in ('ward', 'kmeans', 'treeward'):
        params['n_clusters'] = min(n_clusters, len(offers))
    elif method == 'gmm':
        params['n_components'] = min(n_clusters, len(offers))

    print(f"[Clustering] Running {method.upper()}")

    if method == 'ward':
        model = AgglomerativeClustering(**params)
    elif method == 'kmeans':
        model = KMeans(**params)
    elif method == 'gmm':
        model = GaussianMixture(**params)
    elif method == 'dbscan':
        model = DBSCAN(**params)
    else:
        raise ValueError(f"Unsupported CLUSTER_METHOD: {config.CLUSTER_METHOD}")

    labels = model.fit_predict(X)
    print(f"[Clustering] {method.upper()} completed")

    # DBSCAN labels noise as -1, so build buckets from the actual label set
    unique_labels = sorted(set(labels))

    clustered_offers = {lab: [] for lab in unique_labels}
    for offer, lab in zip(offers, labels):
        clustered_offers[lab].append(offer)

    # Return clusters in label-order; labels array unchanged
    clusters_in_order = [clustered_offers[lab] for lab in unique_labels]
    return clusters_in_order, labels

def aggregate_clusters(clustered_offers):
    """
    Aggregates FlexOffers and DFOs per cluster
    """
    aggregated_offers = []

    for cluster in clustered_offers:
        flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
        dfos = [o for o in cluster if isinstance(o, DFO)]

        if flexoffers and config.TYPE == 'FO':
            if config.ALIGNMENT.lower() == 'balance':
                # from C++
                afo = balance_alignment_aggregate(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
            elif config.ALIGNMENT.lower() == 'balance_fast':
                afo = balance_alignment_tree_merge(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
            else:
                afo = start_alignment_fast(flexoffers)
        if dfos and config.TYPE == 'DFO':
            if config.PARALLEL_CLUSTER_AGGREGATION:
                return aggregate_clusters_parallel_dfo(clustered_offers, numsamples=4, n_jobs=MAX_JOBS)
            else:
                afo = aggnto1(cluster, 4)
        aggregated_offers.append(afo)
    return aggregated_offers


def cluster_and_aggregate_flexoffers(offers, n_clusters):
    """
    Full pipeline: cluster, evaluate, and aggregate.
    """

    if not offers:
        return []

    if config.DYNAMIC_CLUSTERING:
        # search k from CLUSTER_K_MIN to CLUSTER_K_MAX (but < number of offers)
        best_k = None
        best_score = -inf if config.CLUSTER_SELECTION_METRIC == "silhouette" else inf
        for k in range(config.CLUSTER_K_MIN, min(config.CLUSTER_K_MAX, len(offers) - 1) + 1):
            _, labels_k = cluster_offers(offers, n_clusters=k)
            metrics = evaluate_clustering(offers, labels_k)
            sil = metrics["Silhouette Score"]
            db = metrics["Davies-Bouldin Index"]
            if config.CLUSTER_SELECTION_METRIC == "silhouette" and sil is not None:
                if sil > best_score:
                    best_score, best_k = sil, k
        # fallback if nothing found
        if best_k is None:
            best_k = n_clusters
        print(f"[Dynamic Clustering] selected k={best_k} by {config.CLUSTER_SELECTION_METRIC}={best_score:.3f}")
        clustered_flexoffers, labels = cluster_offers(offers, n_clusters=best_k)
    else:
        clustered_flexoffers, labels = cluster_offers(offers, n_clusters=n_clusters)

    # Compute clustering quality metrics
    evaluation = evaluate_clustering(offers, labels)

    print("\n===== Clustering Quality Metrics =====")
    print(f"Silhouette Score: {evaluation['Silhouette Score']:.3f}" if evaluation['Silhouette Score'] is not None else "Silhouette Score: N/A (only 1 cluster)")
    print(f"Davies-Bouldin Index: {evaluation['Davies-Bouldin Index']:.3f}" if evaluation['Davies-Bouldin Index'] is not None else "Davies-Bouldin Index: N/A (only 1 cluster)")
    print("======================================")

    if config.PARALLEL_CLUSTER_AGGREGATION:
        aggregated_fos = aggregate_clusters_parallel(clustered_flexoffers, num_candidates=5, n_jobs=MAX_JOBS)
    else:
        aggregated_fos = aggregate_clusters(clustered_flexoffers)
    return aggregated_fos


def meets_market_compliance(offer: Flexoffer) -> bool:
    if offer.get_min_overall_alloc() < config.MIN_BID_SIZE:
        return True
    return False


def needed_for_compliance(offer):
    if offer.get_total_energy() < config.MIN_BID_SIZE:
        additional_energy_needed = config.MIN_BID_SIZE - offer.get_min_overall_alloc()
        print(additional_energy_needed)
    if config.REQUIRE_UNIFORM:
        avg_energy = sum(offer.get_scheduled_allocation()) / offer.get_duration()
        print(avg_energy)
    return offer


def aggregate_cluster(c, num_candidates):
    print(f"[INFO] Aggregating cluster of size {len(c)} in process {os.getpid()}")

    # Separate FO vs DFO
    flexoffers = [o for o in c if isinstance(o, Flexoffer)]
    dfos = [o for o in c if isinstance(o, DFO)]

    # DFO aggregation shortcut
    if dfos and config.TYPE == 'DFO':
        return aggnto1(dfos, num_candidates)

    # FO aggregation according to configured alignment
    mode = config.ALIGNMENT.lower()
    if mode == 'start':
        return start_alignment_fast(flexoffers)
    elif mode == 'balance':
        return balance_alignment_aggregate(flexoffers, num_candidates)
    elif mode == 'balance_fast':
        return balance_alignment_tree_merge(flexoffers, num_candidates)
    else:
        raise ValueError(f"Unsupported ALIGNMENT mode: {config.ALIGNMENT}")


def aggregate_cluster_dfo(cluster, numsamples):
    print(f"[INFO] Aggregating DFO cluster of size {len(cluster)} in process {os.getpid()}")
    return aggnto1(cluster, numsamples)


def aggregate_clusters_parallel_dfo(clusters, numsamples=4, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(aggregate_cluster_dfo)(cluster, numsamples) for cluster in clusters
    )


def aggregate_clusters_parallel(clusters, num_candidates=5, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(aggregate_cluster)(cluster, num_candidates) for cluster in clusters
    )
