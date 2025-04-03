from flexoffer_logic import Flexoffer, TimeSlice, DFO
import time
import random

def generate_test_scenario():
    scenarios = []

    for i in range(2):
        profile = [TimeSlice(0.5, 1.0) for _ in range(3)]
        fo = Flexoffer(
            offer_id=i,
            earliest_start=int(time.time()),
            latest_start=int(time.time()) + 3600,
            end_time=int(time.time()) + 7200,
            profile=profile,
            duration=3,
            min_overall_alloc=1.0,
            max_overall_alloc=3.0
        )
        scenarios.append(fo)


    for i in range(2):
        dfo = DFO()
        dfo.polygons = [[] for _ in range(24)]
        scenarios.append(dfo)

    return scenarios
