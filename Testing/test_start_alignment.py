import pytest
from datetime import datetime, timedelta
from flexoffer_logic import Flexoffer, TimeSlice
from aggregation.alignments import start_alignment_fast
from helpers import dt_to_unix, dt_to_unix_seconds

def create_test_flexoffer(offer_id, est_hour, lst_hour, et_hour, profile):
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    earliest_start = now.replace(hour=est_hour)
    latest_start = now.replace(hour=lst_hour)
    end_time = now.replace(hour=et_hour)
    duration = len(profile)
    min_energy = sum([p[0] for p in profile])
    max_energy = sum([p[1] for p in profile])
    profile_ts = [TimeSlice(min_val, max_val) for (min_val, max_val) in profile]
    return Flexoffer(
        offer_id,
        dt_to_unix(earliest_start),
        dt_to_unix(latest_start),
        dt_to_unix(end_time),
        profile_ts,
        duration,
        min_energy,
        max_energy
    )

def test_aggregate_flexoffers():
    fo1 = create_test_flexoffer(1, 8, 9, 10, [(1.0, 2.0), (1.5, 2.5)])
    fo2 = create_test_flexoffer(2, 9, 10, 11, [(0.5, 1.0), (1.0, 1.5)])
    fo3 = create_test_flexoffer(3, 7, 8, 9, [(2.0, 3.0), (2.0, 3.0)])
    
    aggregated_offer = start_alignment_fast([fo1, fo2, fo3])
    aggregated_offer.print_flexoffer()