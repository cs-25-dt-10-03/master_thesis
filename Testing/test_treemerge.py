import pytest
import datetime
from flexoffer_logic import Flexoffer, TimeSlice, balance_alignment_aggregate, balance_alignment_tree_merge

# === Conceptual Foundation ===
# We test that the new tree-based merge (balance_alignment_tree_merge)
# produces the same or better absolute balance (sum of absolute average power)
# as the original greedy merge (balance_alignment_aggregate)

# === Mathematical Concept: Absolute Balance ===
# For a FlexOffer fo, with time slices {t1, t2, ..., tn}, each with
# bounds [min_power, max_power], the average power for a slot is:
#     avg_power = (min_power + max_power) / 2
# The absolute balance is:
#     sum(abs(avg_power_t)) for all t
# This is used to measure "how skewed" the profile is from being flat
# Lower is better for scheduling.

# === Computer Science Concept: Tree-based Merging ===
# The tree-based merge builds a binary tree where each node is a merge of two FlexOffers
# This is a divide-and-conquer approach that leads to log(N) depth
# versus a greedy sequential merge that grows linearly

# === Units ===
# Power: kW, Energy: kWh (per slot, assumed 1h here for simplicity)
# Time: Unix timestamps in seconds


def create_mock_flexoffer(offer_id, est_hour, duration, powers):
    """
    Creates a FlexOffer with fixed start hour, duration, and constant power bounds.
    """
    now = datetime.datetime(2025, 1, 1, est_hour, 0)
    est = int(now.timestamp())
    lst = est + 3600 * 2  # allow 2-hour window
    et  = est + duration * 3600
    profile = [TimeSlice(min_p, max_p) for (min_p, max_p) in powers]
    min_total = sum(min_p for (min_p, _) in powers)
    max_total = sum(max_p for (_, max_p) in powers)
    return Flexoffer(offer_id, est, lst, et, profile, duration, min_total, max_total)


def test_tree_merge_equivalence():
    """
    Compare sequential vs. tree-based aggregation on same input
    and assert output profile has similar or better balance.
    """
    fos = [
        create_mock_flexoffer(1, 8, 2, [(1.0, 2.0), (1.0, 2.0)]),
        create_mock_flexoffer(2, 8, 2, [(0.5, 1.5), (0.5, 1.5)]),
        create_mock_flexoffer(3, 9, 2, [(1.2, 2.2), (1.0, 2.0)]),
        create_mock_flexoffer(4, 9, 2, [(0.8, 1.8), (0.9, 1.9)])
    ]

    agg_seq = balance_alignment_aggregate(fos, 5)
    agg_tree = balance_alignment_tree_merge(fos, 5)

    def abs_balance(fo):
        return sum(abs((ts.min_power + ts.max_power) / 2) for ts in fo.get_profile())

    b_seq = abs_balance(agg_seq)
    b_tree = abs_balance(agg_tree)

    print("Sequential merge abs balance:", b_seq)
    print("Tree merge abs balance:", b_tree)

    # Tree merge should be no worse than greedy merge
    assert b_tree <= b_seq + 1e-6  # small epsilon for floating-point rounding


def test_merge_identity_single_input():
    """
    Tree merge with a single FO should return the same offer.
    """
    f = create_mock_flexoffer(42, 10, 2, [(1.0, 1.5), (1.5, 2.0)])
    out = balance_alignment_tree_merge([f], 5)

    assert out.get_duration() == f.get_duration()
    assert len(out.get_profile()) == f.get_duration()
    for orig_ts, out_ts in zip(f.get_profile(), out.get_profile()):
        assert abs(orig_ts.min_power - out_ts.min_power) < 1e-6
        assert abs(orig_ts.max_power - out_ts.max_power) < 1e-6


def test_aggregation_correctness():
    """
    Check that aggregated FlexOffer has expected profile length and energy bounds
    when aggregating multiple offers.
    """
    f1 = create_mock_flexoffer(201, 7, 2, [(1.0, 2.0), (2.0, 3.0)])
    f2 = create_mock_flexoffer(202, 7, 2, [(0.5, 1.0), (1.0, 1.5)])
    f3 = create_mock_flexoffer(203, 7, 2, [(0.2, 0.8), (0.3, 0.9)])

    agg = balance_alignment_tree_merge([f1, f2, f3], 1)
    profile = agg.get_profile()

    assert len(profile) >= 2
    # Check that powers add up correctly
    expected = [
        (1.0 + 0.5 + 0.2, 2.0 + 1.0 + 0.8),
        (2.0 + 1.0 + 0.3, 3.0 + 1.5 + 0.9)
    ]
    for ts, (emin, emax) in zip(profile, expected):
        assert abs(ts.min_power - emin) < 1e-6
        assert abs(ts.max_power - emax) < 1e-6

    # Check that total energy matches
    total_min = 1.0 + 2.0 + 0.5 + 1.0 + 0.2 + 0.3
    total_max = 2.0 + 3.0 + 1.0 + 1.5 + 0.8 + 0.9
    assert abs(agg.get_min_overall_alloc() - total_min) < 1e-6
    assert abs(agg.get_max_overall_alloc() - total_max) < 1e-6



def test_aggregation_with_misalignment():
    """
    Aggregates misaligned FlexOffers that still overlap. Tests correctness of profile and timing.
    """
    f1 = create_mock_flexoffer(301, 6, 2, [(1.0, 2.0), (1.5, 2.5)])  # Starts at 6:00
    f2 = create_mock_flexoffer(302, 7, 2, [(0.5, 1.0), (0.8, 1.2)])  # Starts at 7:00
    f3 = create_mock_flexoffer(303, 7, 2, [(0.2, 0.6), (0.4, 0.9)])  # Also starts at 7:00

    agg = balance_alignment_tree_merge([f1, f2, f3], 3)
    profile = agg.get_profile()

    # Check profile is long enough to accommodate the offset merge
    assert len(profile) >= 4  # Offset 1 hour + 2 durations = 4 slots expected

    # Verify time alignment makes sense
    est = agg.get_est()
    lst = agg.get_lst()
    et = agg.get_et()

    assert est <= min(f.get_est() for f in [f1, f2, f3])
    assert lst >= max(f.get_lst() for f in [f1, f2, f3])
    assert et >= max(f.get_et() for f in [f1, f2, f3])

    # Optional: print for debug
    print("Aggregated EST:", datetime.datetime.fromtimestamp(est))
    print("Aggregated LST:", datetime.datetime.fromtimestamp(lst))
    print("Aggregated ET:", datetime.datetime.fromtimestamp(et))
