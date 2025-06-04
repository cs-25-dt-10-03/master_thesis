import time
from datetime import datetime, timedelta
from flexoffer_logic import Flexoffer, TimeSlice, start_alignment_aggregate, balance_alignment_aggregate, balance_alignment_tree_merge
from evaluation.utils.plot_flexOffer import plot_flexoffer_aggregation

# Time utilities
def dt_to_unix(dt):
    return int(dt.timestamp())

now = datetime.now().replace(minute=0, second=0, microsecond=0)
est1 = dt_to_unix(now + timedelta(hours=2))
lst1 = dt_to_unix(now + timedelta(hours=6))
et1  = dt_to_unix(now + timedelta(hours=9))

est2 = dt_to_unix(now + timedelta(hours=0))
lst2 = dt_to_unix(now + timedelta(hours=6))
et2  = dt_to_unix(now + timedelta(hours=9))

# Define simple power profiles
profile1 = [TimeSlice(0.0, 11.0) for _ in range((et1 - lst1)//3600)]
profile2 = [TimeSlice(0.0, 11.0) for _ in range((et2 - lst2)//3600)]

# Create FlexOffers
fo1 = Flexoffer(
    offer_id=1,
    earliest_start=est1,
    latest_start=lst1,
    end_time=et1,
    profile=profile1,
    duration=len(profile1),
    min_overall_alloc=5.0,
    max_overall_alloc=10.0
)

fo2 = Flexoffer(
    offer_id=2,
    earliest_start=est2,
    latest_start=lst2,
    end_time=et2,
    profile=profile2,
    duration=len(profile2),
    min_overall_alloc=6.0,
    max_overall_alloc=9.0
)

# Aggregate
fo_agg = start_alignment_aggregate([fo1, fo2])
fo_agg_flex = balance_alignment_aggregate([fo1, fo2], 10)

# Plot
plot_flexoffer_aggregation(fo1, fo2, fo_agg, resolution_seconds=3600)
plot_flexoffer_aggregation(fo1, fo2, fo_agg_flex, resolution_seconds=3600)