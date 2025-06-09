# clustering/feature_factory.py
import numpy as np
from config import config
from flexoffer_logic import Flexoffer, DFO

def extract_time_range(offer):
    return np.array([offer.get_est(), offer.get_lst()])

def extract_energy_profile(offer):
    if config.TYPE == "DFO":
        return np.array([offer.get_min_total_energy(), offer.get_max_total_energy()])
    elif config.TYPE == "FO":
        return np.array([offer.get_min_overall_alloc(), offer.get_max_overall_alloc()]) 

def extract_full(offer):
    if config.TYPE == "FO":
        return np.array([
            offer.get_est(),
            offer.get_lst(),
            offer.get_duration(),
            offer.get_et(),
        ])
    
# 2) Build a dict to map config keys → functions 
FEATURE_EXTRACTORS = {
    'time_range':    extract_time_range,
    'energy':        extract_energy_profile,
    'full':       extract_full
}

def extract_features(offer):
    key = config.CLUSTER_FEATURE_SET
    return FEATURE_EXTRACTORS[key](offer)
