import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flexoffer_logic import Flexoffer, TimeSlice

def create_aggregated_offers(min_lists, max_lists, start_time):
    offers = []
    for mins, maxs in zip(min_lists, max_lists):
        profile = [TimeSlice(lo, hi) for lo,hi in zip(mins, maxs)]
        fo = Flexoffer(
            0,                    # offer_id
            start_time,           # earliest_start
            len(profile)-1,       # latest_start
            len(profile),         # end_time
            profile,              # profile: list of TimeSlice
            len(profile),         # duration
            float(sum(mins)),     # min_overall_alloc
            float(sum(maxs))      # max_overall_alloc
        )
        offers.append(fo)
    return offers
