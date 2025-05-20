import pytest
from flexoffer_logic import Flexoffer, TimeSlice, balance_alignment_aggregate
import time

def make_flexoffer(offer_id, est_offset_slots, duration_slots, resolution, power_range, est_base=None):
    if est_base is None:
        est_base = int(time.time())
    est = est_base + est_offset_slots * resolution
    lst = est + 2 * resolution
    et = lst + duration_slots * resolution

    profile = [TimeSlice(power_range[0], power_range[1]) for _ in range(duration_slots)]
    min_energy = 5.0
    max_energy = 70.0

    return Flexoffer(offer_id, est, lst, et, profile, duration_slots, min_energy, max_energy)


def test_balance_alignment_diverse_offsets():
    resolution = 900
    est_base = 1600000000

    fo1 = make_flexoffer(1, est_offset_slots=0, duration_slots=4, resolution=resolution, power_range=(1, 2), est_base=est_base)
    fo2 = make_flexoffer(2, est_offset_slots=2, duration_slots=4, resolution=resolution, power_range=(2, 3), est_base=est_base)
    fo3 = make_flexoffer(3, est_offset_slots=4, duration_slots=4, resolution=resolution, power_range=(0.5, 1.5), est_base=est_base)

    aggregated = balance_alignment_aggregate([fo1, fo2, fo3], num_candidates=5)

    assert aggregated.get_est() == min(fo.get_est() for fo in [fo1, fo2, fo3])

    expected_lst = aggregated.get_est() + min(fo.get_lst() - fo.get_est() for fo in [fo1, fo2, fo3])
    assert aggregated.get_lst() == expected_lst

    offsets = [(fo.get_est() - aggregated.get_est()) // resolution for fo in [fo1, fo2, fo3]]
    expected_duration = max(offset + fo.get_duration() for offset, fo in zip(offsets, [fo1, fo2, fo3]))
    assert aggregated.get_duration() == expected_duration
    assert len(aggregated.get_profile()) == expected_duration

    expected_et = aggregated.get_lst() + aggregated.get_duration() * resolution
    assert aggregated.get_et() == expected_et

    agg_profile = aggregated.get_profile()
    for t in range(expected_duration):
        expected_min = 0.0
        expected_max = 0.0
        for fo in [fo1, fo2, fo3]:
            offset = (fo.get_est() - aggregated.get_est()) // resolution
            if 0 <= t - offset < fo.get_duration():
                ts = fo.get_profile()[t - offset]
                expected_min += ts.min_power
                expected_max += ts.max_power
        assert abs(agg_profile[t].min_power - expected_min) < 1e-6, f"Min power mismatch at slot {t}"
        assert abs(agg_profile[t].max_power - expected_max) < 1e-6, f"Max power mismatch at slot {t}"

    profile_min_energy = sum(ts.min_power for ts in agg_profile)
    profile_max_energy = sum(ts.max_power for ts in agg_profile)
    assert aggregated.get_min_overall_alloc() <= profile_max_energy + 1e-6
    assert aggregated.get_max_overall_alloc() >= profile_min_energy - 1e-6
    