import json
import numpy as np
import os
import flexoffer_logic 

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


class config:
    #SIMULATION
    NUM_EVS = 20
    NUM_CLUSTERS = 10
    SIMULATION_DAYS = 10
    SIMULATION_START_DATE = "2024-01-16"
    USE_SYNTHETIC = True
    TYPE = 'FO'

    #MARKET
    TIME_RESOLUTION = 3600 
    PENALTY = 1000
    RESOLUTION = "hourly" if TIME_RESOLUTION == 3600 else "15min"
    DATA_FILEPATH = os.path.join("..", "SmartCharging_2020_to_2032")

    # valid values: "joint", "sequential_reserve_first"
    MODE = "sequential_reserve_first"
    RUN_SPOT = True
    RUN_RESERVE = False
    RUN_ACTIVATION = False


    #CLUSTERING 
    CLUSTER_FEATURE_SET = 'time_range'
    CLUSTER_DISTANCE_METRIC = 'euclidean'
    CLUSTER_LINKAGE = 'complete'
    USE_THRESHOLD = True

    MIN_CLUSTER_ENERGY = None
    MAX_CLUSTER_ENERGY = None

    CLUSTER_THRESHOLD_SET    = 'time_range'

    CLUSTER_THRESHOLDS = {
        'time_range': [7200, 7200],
        'energy':     [50.0],
    }


    ALIGNMENT = "start"
    CLUSTER_METHOD = 'ward'
    CLUSTER_PARAMS = {
        'ward':   {'n_clusters': NUM_CLUSTERS},
        'kmeans': {'n_clusters': NUM_CLUSTERS, 'random_state': 42},
        'gmm':    {'n_components': NUM_CLUSTERS, 'covariance_type': 'full', 'random_state': 42},
        'dbscan': {'eps': 0.5, 'min_samples': 5}
    }

    #DYNAMIC
    DYNAMIC_CLUSTERING     = True
    CLUSTER_K_MIN          = 2
    CLUSTER_K_MAX          = 20
    CLUSTER_SELECTION_METRIC = 'silhouette'

    @classmethod
    def apply_override(cls, overrides):
        for k, v in overrides.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
        flexoffer_logic.set_time_resolution(cls.TIME_RESOLUTION)
