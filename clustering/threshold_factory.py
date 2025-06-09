import numpy as np
from config import config
from flexoffer_logic import Flexoffer, DFO

def extract_time_range(offer):
    # [EST, LST]
    return np.array([offer.get_est(), offer.get_lst()])

def extract_energy(offer):
    if config.TYPE == "DFO":
        return np.array([offer.get_min_total_energy(), offer.get_max_total_energy()])
    elif config.TYPE == "FO":
        return np.array([offer.get_min_alloc(), offer.get_max_alloc()]) 


THRESHOLD_EXTRACTORS = {
    'time_range': extract_time_range,
    'energy':     extract_energy,
}

def get_threshold_features(offer):
    key = config.CLUSTER_THRESHOLD_SET
    fn = THRESHOLD_EXTRACTORS[key]
    return fn(offer)

def get_threshold_values():
    key = config.CLUSTER_THRESHOLD_SET
    return np.array(config.CLUSTER_THRESHOLDS[key], dtype=float)
