import pytest
from datetime import datetime, timedelta
from classes.flexOffer import FlexOffer
from config import config
from aggregation.alignments import start_alignment_fast

def create_test_flexoffer(offer_id, start_hour, duration_hours, profile):
    earliest_start = datetime.now().replace(hour=start_hour, minute=0, second=0, microsecond=0)
    latest_start = earliest_start + timedelta(hours=1)
    end_time = earliest_start + timedelta(hours=duration_hours)
    duration = timedelta(hours=duration_hours)
    min_energy = sum([p[0] for p in profile])
    max_energy = sum([p[1] for p in profile])

    return FlexOffer(offer_id, earliest_start, latest_start, end_time, duration, profile, min_energy, max_energy)

def test_start_alignment():
    fo1 = create_test_flexoffer(1, 8, 3, [(1, 2), (1.5, 2.5), (2, 3)])
    fo2 = create_test_flexoffer(2, 9, 2, [(0.5, 1), (1, 1.5)])
    fo3 = create_test_flexoffer(3, 7, 4, [(2, 3), (2, 3), (2, 3), (2, 3)])
    
    aggregated_offer = start_alignment_fast([fo1, fo2, fo3])
    
    assert len(aggregated_offer.energy_profile) == (aggregated_offer.duration.seconds // config.TIME_RESOLUTION)

    print("Aggregated FlexOffer:", aggregated_offer)
