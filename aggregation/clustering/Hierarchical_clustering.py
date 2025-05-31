import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from time import time
from sklearn.preprocessing import StandardScaler
from config import config
from math import inf
from aggregation.alignments import start_alignment_fast
from flexoffer_logic import Flexoffer, DFO, aggnto1, balance_alignment_aggregate, balance_alignment_tree_merge
from aggregation.clustering.metrics import evaluate_clustering


def extract_features(offer):
    """
    Creates a feature vector for an offer.
    """
    if isinstance(offer, Flexoffer):
        return config.define_clustering_features_fo(offer)
    elif isinstance(offer, DFO):
        return config.define_clustering_features_dfo(offer)
    else:
        raise ValueError("Unknown offer type")


def cluster_offers_on_X(X, offers, n_clusters):
    """
    Clusters using precomputed features X.
    Returns clusters list, labels, and clustering_time.
    """
    print(f"[Clustering] Running on precomputed X for {len(offers)} offers with k={n_clusters}...")
    start_time = time()

    method = config.CLUSTER_METHOD.lower()
    params = config.CLUSTER_PARAMS.get(method, {})
    if method in ('ward', 'kmeans'):
        params['n_clusters'] = min(n_clusters, len(offers))
    elif method == 'gmm':
        params['n_components'] = min(n_clusters, len(offers))

    print(f"[Clustering] Method: {method.upper()}, Params: {params}")

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

    t0 = time()
    labels = model.fit_predict(X)
    t1 = time()
    clustering_time = t1 - t0
    print(f"[Timing] {method.upper()} fit_predict: {clustering_time:.3f}s")

    unique_labels = sorted(set(labels))
    clusters = [[] for _ in unique_labels]
    for offer, lab in zip(offers, labels):
        clusters[lab].append(offer)

    total_time = time() - start_time
    print(f"[Clustering] Completed {method.upper()} in {total_time:.3f}s total")
    return clusters, labels, clustering_time


def aggregate_clusters(clustered_offers):
    """
    Aggregates FlexOffers and DFOs per cluster.
    """
    aggregated = []
    print(f"[Aggregation] Aggregating {len(clustered_offers)} clusters...")
    t0 = time()

    for idx, cluster in enumerate(clustered_offers):
        t1 = time()
        flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
        dfos = [o for o in cluster if isinstance(o, DFO)]

        if flexoffers and config.TYPE == 'FO':
            mode = config.ALIGNMENT.lower()
            if mode == 'balance':
                agg_offer = balance_alignment_aggregate(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
            elif mode == 'balance_fast':
                agg_offer = balance_alignment_tree_merge(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
            else:
                agg_offer = start_alignment_fast(flexoffers)
        elif dfos and config.TYPE == 'DFO':
            agg_offer = aggnto1(cluster, 4)
        else:
            continue

        t2 = time()
        print(f"[Timing] Cluster {idx} size {len(cluster)} aggregated in {t2 - t1:.3f}s")
        aggregated.append(agg_offer)

    t3 = time()
    agg_time = t3 - t0
    print(f"[Timing] Total aggregation: {agg_time:.3f}s")
    return aggregated, agg_time


def cluster_and_aggregate_flexoffers(offers, n_clusters):
    """
    Full pipeline: cluster (with evaluation) and aggregate.
    Returns aggregated offers, clustering_time, aggregation_time.
    """
    if not offers:
        return [], 0.0, 0.0

    print("[Pipeline] Starting cluster_and_aggregate_flexoffers...")
    t_pipeline = time()

    # 1) Precompute features + scale
    print("[Pipeline] Precomputing features & scaling...")
    t0 = time()
    feats = [extract_features(o) for o in offers]
    X = np.vstack(feats)
    X = StandardScaler().fit_transform(X)
    t1 = time()
    print(f"[Timing] Feature+scaling: {t1 - t0:.3f}s")

    # 2) Clustering
    if config.DYNAMIC_CLUSTERING:
        best_k = None
        best_score = -inf if config.CLUSTER_SELECTION_METRIC == 'silhouette' else inf
        for k in range(config.CLUSTER_K_MIN, min(config.CLUSTER_K_MAX, len(offers)-1) + 1):
            _, labels_k, timing_k = cluster_offers_on_X(X, offers, n_clusters=k)
            metrics = evaluate_clustering(offers, labels_k)
            sil = metrics.get('Silhouette Score')
            if config.CLUSTER_SELECTION_METRIC == 'silhouette' and sil is not None:
                if sil > best_score:
                    best_score, best_k = sil, k
        if best_k is None:
            best_k = n_clusters
        print(f"[Dynamic] Selected k={best_k} (score={best_score:.3f})")
        clusters, labels, clustering_time = cluster_offers_on_X(X, offers, n_clusters=best_k)
    else:
        clusters, labels, clustering_time = cluster_offers_on_X(X, offers, n_clusters=n_clusters)

    # Evaluate clustering metrics
    t_eval0 = time()
    evaluation = evaluate_clustering(offers, labels)
    t_eval1 = time()
    print(f"[Timing] Evaluation: {t_eval1 - t_eval0:.3f}s")
    print("\n===== Clustering Metrics =====")
    sil_val = evaluation.get('Silhouette Score')
    db_val = evaluation.get('Davies-Bouldin Index')
    if sil_val is not None:
        print(f"Silhouette Score: {sil_val:.3f}")
    else:
        print("Silhouette Score: N/A")
    if db_val is not None:
        print(f"Davies-Bouldin Index: {db_val:.3f}")
    else:
        print("Davies-Bouldin Index: N/A")
    print("=============================")

    # 3) Aggregation
    aggregated, aggregation_time = aggregate_clusters(clusters)

    total_time = time() - t_pipeline
    print(f"[Pipeline] Total time: {total_time:.3f}s")
    return aggregated, clustering_time, aggregation_time
