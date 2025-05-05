import pytest
import pandas as pd
import numpy as np
from config import config
from flexoffer_logic import Flexoffer, TimeSlice, DFO
from aggregation.clustering.Hierarchical_clustering import cluster_and_aggregate_flexoffers
from optimization.scheduler import schedule_offers
from evaluation.evaluation_pipeline import greedy_baseline_schedule, compute_profit

# Helper to build a simple Flexoffer

def make_flexoffer(est_hour, lst_hour, duration_slots, max_power_kw, total_energy_kwh):
    # est_hour, lst_hour are ints representing hours since epoch for simplicity
    est_ts = est_hour * 3600
    lst_ts = lst_hour * 3600
    # profile: each slot has same min/max power
    profile = [TimeSlice(0.0, max_power_kw) for _ in range(duration_slots)]
    fo = Flexoffer(
        1,
        est_ts,
        lst_ts,
        lst_ts + duration_slots * 3600,
        profile,
        duration_slots,
        total_energy_kwh,
        total_energy_kwh * 1.5
    )
    return fo

@pytest.fixture
def simple_offers():
    # Two offers starting at hour 0, two slots, max_power 1 kW, total 1 kWh each
    fo1 = make_flexoffer(est_hour=0, lst_hour=0, duration_slots=2, max_power_kw=1.0, total_energy_kwh=1.0)
    fo2 = make_flexoffer(est_hour=0, lst_hour=0, duration_slots=2, max_power_kw=1.0, total_energy_kwh=1.0)
    return [fo1, fo2]

@pytest.fixture
def price_series():
    spot = pd.Series([10.0, 5.0, 1.0])
    # reserve and activation as empty structures
    reserve = pd.DataFrame(columns=["up", "dn"]).reindex(range(3), fill_value=0.0)
    activation = pd.DataFrame(columns=["up", "dn"]).reindex(range(3), fill_value=0.0)
    indicators = pd.DataFrame()  # unused
    return spot, reserve, activation, indicators


def test_greedy_baseline_earliest_fill(simple_offers, price_series):
    spot, reserve, activation, indicators = price_series
    sol = greedy_baseline_schedule(simple_offers)
    # Each offer has 1 kWh total, max_power 1 kW, dt=1h -> 1 slot filled
    # All slots start at idx 0 => both filled slot 0
    assert sol["p"][0] == {0: 1.0}
    assert sol["p"][1] == {0: 1.0}
    # compute profit: spot_rev = (1+1)*spot[0]*1h = 2*10 = 20
    profit = compute_profit(sol, spot, reserve, activation, indicators)
    assert pytest.approx(profit["spot_rev"]) == 20.0
    # no reserve/activation revenue
    assert profit["res_rev"] == 0.0
    assert profit["act_rev"] == 0.0
    # total_rev = -spot_rev
    assert pytest.approx(profit["total_rev"]) == -20.0


def test_clustering_aggregation_identity(simple_offers):
    # With two identical offers, clustering into 1 cluster should aggregate properly
    # Force NUM_CLUSTERS=1 for test
    config.NUM_CLUSTERS = 1
    agg = cluster_and_aggregate_flexoffers(simple_offers, config.NUM_CLUSTERS)
    # We expect exactly 1 aggregated Flexoffer returned
    assert len(agg) == 1
    # Aggregated profile power should be sum: 1+1 = 2 kW in each slot
    profile = agg[0].get_profile()
    assert len(profile) == simple_offers[0].get_duration()
    for ts in profile:
        assert pytest.approx(ts.max_power) == 2.0

def test_clustering_aggregation_identity(simple_offers):
    config.NUM_CLUSTERS = 1
    agg = cluster_and_aggregate_flexoffers(simple_offers, config.NUM_CLUSTERS)
    assert len(agg) == 1
    profile = agg[0].get_profile()
    assert len(profile) == simple_offers[0].get_duration()
    for ts in profile:
        assert pytest.approx(ts.max_power) == 2.0



def test_overall_pipeline_improves_or_equals_baseline(simple_offers, price_series):
    spot, reserve, activation, indicators = price_series
    # Evaluate both baseline and scheduler over simple offers
    base_sol = greedy_baseline_schedule(simple_offers)
    base_profit = compute_profit(base_sol, spot, reserve, activation, indicators)
    # Use scheduler on aggregated offers
    config.NUM_CLUSTERS = 1
    agg = cluster_and_aggregate_flexoffers(simple_offers, config.NUM_CLUSTERS)
    sched_sol = schedule_offers(agg, spot_prices=spot, reserve_prices=reserve, activation_prices=activation, indicators=indicators)
    sched_profit = compute_profit(sched_sol, spot, reserve, activation, indicators)
    # Scheduler should be at least as good as greedy baseline
    print(f"sch_total: {sched_profit["total_rev"]}")
    print(f"base_total: {base_profit["total_rev"]}")

    assert sched_profit["total_rev"] >= base_profit["total_rev"]


# Testing/test_dfo_aggregation.py

import pytest
from flexoffer_logic import DFO, aggnto1

def make_simple_dfo(min_prev, max_prev, earliest_start, numsamples=5, charging_power=1.0):
    """
    Construct a DFO with given min_prev/max_prev vectors and earliest_start,
    then generate its dependency polygons and compute its latest_start.
    """
    dfo = DFO(0, min_prev, max_prev, numsamples, charging_power, sum(min_prev), sum(max_prev), earliest_start)
    # build the full dependency graph
    dfo.generate_dependency_polygons()
    dfo.calculate_latest_start_time()
    return dfo

def test_aggnto1_sums_min_max_totals():
    # Create two simple DFOs with distinct min/max total energies
    d1_min_prev = [1.0, 0.5, 0.0]
    d1_max_prev = [0.0, 0.5, 1.0]
    d2_min_prev = [2.0, 1.0]      # shorter window
    d2_max_prev = [0.0, 2.0]

    # Use the same earliest_start for simplicity
    start_ts = 0

    d1 = make_simple_dfo(d1_min_prev, d1_max_prev, earliest_start=start_ts)
    d2 = make_simple_dfo(d2_min_prev, d2_max_prev, earliest_start=start_ts)

    # The explicit totals we passed in:
    expected_min = d1.min_total_energy + d2.min_total_energy
    expected_max = d1.max_total_energy + d2.max_total_energy

    # Aggregate via the Python binding
    agg = aggnto1([d1, d2], numsamples=5)

    # Check that the aggregated DFO carries the correct sums
    assert agg.min_total_energy == pytest.approx(expected_min), (
        f"Expected aggregated.min_total_energy={expected_min}, got {agg.min_total_energy}"
    )
    assert agg.max_total_energy == pytest.approx(expected_max), (
        f"Expected aggregated.max_total_energy={expected_max}, got {agg.max_total_energy}"
    )

def test_multi_aggnto1_three_dfos():
    # And test chaining 3 DFOs
    d0 = make_simple_dfo([0.5,0.0], [1.0,0.5], earliest_start=0)
    d1 = make_simple_dfo([1.5,0.5], [2.0,1.0], earliest_start=0)
    d2 = make_simple_dfo([0.0,1.0,0.5], [0.5,1.5,1.0], earliest_start=0)

    expected_min = d0.min_total_energy + d1.min_total_energy + d2.min_total_energy
    expected_max = d0.max_total_energy + d1.max_total_energy + d2.max_total_energy

    agg = aggnto1([d0, d1, d2], numsamples=5)

    assert agg.min_total_energy == pytest.approx(expected_min)
    assert agg.max_total_energy == pytest.approx(expected_max)



from flexoffer_logic import Flexoffer, TimeSlice
from config import config
import numpy as np


def value_weighted_aggregate(offers, spot_prices, reserve_prices=None, activation_prices=None):
    """
    Aggregates a list of FlexOffers into one FlexOffer, ordering each offer's slots by
    a combined multi-market value and then bundling kW capacities by rank.

    Parameters:
        offers: List[Flexoffer]
        spot_prices: pd.Series indexed by slot
        reserve_prices: pd.DataFrame with 'up','dn' columns, optional
        activation_prices: pd.DataFrame with 'up','dn' columns, optional

    Returns:
        aggregated Flexoffer capturing combined flexibility across markets.
    """
    # Resolution and timing
    slot_sec = config.TIME_RESOLUTION  # seconds per slot
    # reference time
    t0 = min(fo.get_est() for fo in offers)

    # Determine the maximum duration across offers
    durations = [fo.get_duration() for fo in offers]
    max_dur = max(durations)

    # Build value arrays for each offer
    value_arrays = []
    for fo in offers:
        est = fo.get_est()
        dur = fo.get_duration()
        v = []
        for i in range(dur):
            # global slot index
            slot_idx = int((est - t0) / slot_sec) + i
            # spot
            sp = spot_prices.iloc[slot_idx]
            # reserve
            ru = rd = 0.0
            if reserve_prices is not None:
                ru, rd = reserve_prices.iloc[slot_idx]
            # activation
            au = ad = 0.0
            if activation_prices is not None:
                au, ad = activation_prices.iloc[slot_idx]
            # combined value: reward reserve/activation, penalize spot cost
            val = -sp + ru + rd + au + ad
            v.append(val)
        # pad shorter arrays with the lowest value
        if dur < max_dur:
            pad_val = min(v)
            v.extend([pad_val] * (max_dur - dur))
        value_arrays.append(np.array(v))

    # For each offer, get the ranking of slots by descending value
    rank_indices = [np.argsort(-v) for v in value_arrays]

    # Build aggregated profile: slice j corresponds to j-th best slots across all offers
    aggregated_profile = []
    for j in range(max_dur):
        # sum the max_power from each offer's j-th best slot
        total_power = 0.0
        for fo, ranks in zip(offers, rank_indices):
            idx = ranks[j]
            prof = fo.get_profile()
            if idx < len(prof):
                total_power += prof[idx].max_power
        aggregated_profile.append(TimeSlice(0.0, total_power))

    # Build the aggregated Flexoffer
    earliest = t0
    latest   = max(fo.get_lst() for fo in offers)
    end_time = earliest + max_dur * slot_sec

    min_total = sum(fo.get_min_overall_alloc() for fo in offers)
    max_total = sum(fo.get_max_overall_alloc() for fo in offers)

    aggregated = Flexoffer(
        id=-1,
        est=earliest,
        lst=latest,
        deadline=end_time,
        profile=aggregated_profile,
        duration=max_dur,
        min_overall_alloc=min_total,
        max_overall_alloc=max_total
    )
    return aggregated
