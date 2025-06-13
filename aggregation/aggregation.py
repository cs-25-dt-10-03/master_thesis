import time
from flexoffer_logic import Flexoffer, DFO, aggnto1, balance_alignment_aggregate, balance_alignment_tree_merge, get_time_resolution
from config import config
from aggregation.alignments import start_alignment_fast


def aggregate_cluster(cluster):
    """
    Aggregates a single cluster of offers into one aggregated offer.
    """
    flexoffers = [o for o in cluster if isinstance(o, Flexoffer)]
    dfos = [o for o in cluster if isinstance(o, DFO)]

    if flexoffers and config.TYPE == 'FO':
        mode = config.ALIGNMENT.lower()
        if mode == 'balance':
            return balance_alignment_aggregate(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
        elif mode == 'balance_fast':
            return balance_alignment_tree_merge(flexoffers, config.CLUSTER_PARAMS.get('balance_candidates', 5))
        else:
            return start_alignment_fast(flexoffers)

    if dfos and config.TYPE == 'DFO':
        return aggnto1(dfos, config.CLUSTER_PARAMS.get('numsamples', 5))

    return None


def aggregate_clusters(clustered_offers):
    """
    Aggregates each cluster of offers and returns a list of aggregated offers.
    """
    aggregated = []
    t_start = time.time()

    for idx, cluster in enumerate(clustered_offers):
        agg_offer = aggregate_cluster(cluster)
        if agg_offer is not None:
            aggregated.append(agg_offer)

    t_end = time.time()
    return aggregated, t_end - t_start
