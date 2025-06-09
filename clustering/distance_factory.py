# clustering/distance_factory.py
from scipy.spatial.distance import euclidean, cityblock
from config import config

DISTANCE_METRICS = {
    'euclidean':    'euclidean',
    'manhattan':    'manhattan', 
}

def get_distance_metric():
    m = config.CLUSTER_DISTANCE_METRIC
    return DISTANCE_METRICS[m]
