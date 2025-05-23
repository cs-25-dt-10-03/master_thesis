import json
import numpy as np
import os
import flexoffer_logic

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    flexoffer_logic.reload_config()  # Loads config into c++
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


class config:
    _config_data = load_config()

    # Simulations
    NUM_EVS = _config_data.get("NUM_EVS", 20)
    NUM_CLUSTERS = 5
    SIMULATION_DAYS = _config_data.get("SIMULATION_DAYS", 30)
    SIMULATION_START_DATE = _config_data.get("SIMULATION_START_DATE", "2024-03-01")
    USE_SYNTHETIC = True

    TYPE = 'FO'
    # Market requirements
    TIME_RESOLUTION = _config_data.get("TIME_RESOLUTION", 3600)
    MIN_BID_SIZE = _config_data.get("MIN_BID_SIZE", 100)
    REQUIRE_UNIFORM = _config_data.get("REQUIRE_UNIFORM", 0)

    PENALTY = 1000
    RESOLUTION = "hourly" if TIME_RESOLUTION == 3600 else "15min"
    DATA_FILEPATH = os.path.join("..", "SmartCharging_2020_to_2032")

    # valid values: "joint", "sequential", "sequential_reserve_first"
    MODE = "sequential_reserve_first"
    RUN_SPOT = True
    RUN_RESERVE = False
    RUN_ACTIVATION = False

    #start, balance, balance_fast
    ALIGNMENT = "start"

    # Which algorithm to use: 'ward', 'kmeans', 'gmm', 'dbscan'
    CLUSTER_METHOD = 'ward'
    # Hyperparameters for each:
    CLUSTER_PARAMS = {
        'ward':   {'n_clusters': NUM_CLUSTERS},
        'kmeans': {'n_clusters': NUM_CLUSTERS, 'random_state': 42},
        'gmm':    {'n_components': NUM_CLUSTERS, 'covariance_type': 'full', 'random_state': 42},
        'dbscan': {'eps': 0.5, 'min_samples': 5}
    }

    DYNAMIC_CLUSTERING     = True
    CLUSTER_K_MIN          = 2
    CLUSTER_K_MAX          = NUM_CLUSTERS
    CLUSTER_SELECTION_METRIC = "silhouette"

    PARALLEL_CLUSTER_AGGREGATION = False
    PARALLEL_N_JOBS = 4

    COST_SENSITIVE_CLUSTERING = False
    COST_FEATURE_WEIGHT       = 1.0


    @classmethod
    def define_clustering_features_fo(cls, offer: flexoffer_logic.Flexoffer):
        return np.array([
            offer.get_est(),
            offer.get_lst(),
        #    offer.get_min_overall_alloc(),
        ])

    @classmethod
    def define_clustering_features_dfo(cls, offer: flexoffer_logic.DFO):
        return np.array([
            offer.get_est(),
            offer.get_lst(),
            offer.min_total_energy,
        ])

    @classmethod
    def apply_override(cls, overrides):
        for k, v in overrides.items():
            if hasattr(cls, k):
                setattr(cls, k, v)
