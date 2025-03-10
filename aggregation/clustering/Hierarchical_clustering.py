import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
from flexoffer_logic import Flexoffer



def extract_features(flex_offers: List[Flexoffer]):
    features = []
    
    for fo in flex_offers:
        est = fo.get_est_hour()
        lst = fo.get_lst_hour()
        time_flexibility = lst - est
        total_energy = fo.get_total_energy()
        
        features.append([est, lst, time_flexibility, total_energy])
    
    return np.array(features)

def cluster_flexoffers(flex_offers, n_clusters=3):
    features = extract_features(flex_offers)

    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clustering.fit_predict(features)

    return labels

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

    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    linkage_matrix = linkage(normalized_features, method=method)

    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=[fo.get_offer_id() for fo in flex_offers], leaf_rotation=90)
    plt.xlabel("FlexOffer ID")
    plt.ylabel("Cluster Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.show()