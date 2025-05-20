import pytest
from flexoffer_logic import Flexoffer, TimeSlice, start_alignment_aggregate
import time
from config import config

def make_flexoffer(offer_id, est_offset, duration, resolution, power_range, est_base=None):
    if est_base is None:
        est_base = int(time.time())
    est = est_base + (est_offset * resolution)
    lst = est + 2 * resolution
    et = lst + duration * resolution

    profile = [TimeSlice(power_range[0], power_range[1]) for _ in range(duration)]
    min_energy = 5.0  # realistic bound
    max_energy = 70.0

    print(f"est defined as: {est}")
    return Flexoffer(offer_id, est, lst, et, profile, duration, min_energy, max_energy)

def test_start_alignment_basic_case():
    resolution = config.TIME_RESOLUTION
    est_base = 1600000000

    fo1 = make_flexoffer(1, 0, 4, resolution, (1, 2), est_base)
    fo2 = make_flexoffer(2, 1, 4, resolution, (2, 3), est_base)
    fo3 = make_flexoffer(3, 2, 4, resolution, (0.5, 1.5), est_base)

    aggregated = start_alignment_aggregate([fo1, fo2, fo3])

    assert aggregated.get_est() == est_base

    expected_lst = est_base + min(
        fo1.get_lst() - fo1.get_est(),
        fo2.get_lst() - fo2.get_est(),
        fo3.get_lst() - fo3.get_est()
    )
    assert aggregated.get_lst() == expected_lst

    expected_duration = max((fo.get_est() - est_base) // resolution + fo.get_duration() for fo in [fo1, fo2, fo3])

    assert aggregated.get_duration() == expected_duration
    assert len(aggregated.get_profile()) == expected_duration

    agg_profile = aggregated.get_profile()
    for t in range(expected_duration):
        expected_min = 0.0
        expected_max = 0.0
        for fo in [fo1, fo2, fo3]:
            offset = (fo.get_est() - est_base) // resolution
            if 0 <= t - offset < fo.get_duration():
                ts = fo.get_profile()[t - offset]
                expected_min += ts.min_power
                expected_max += ts.max_power
        assert abs(agg_profile[t].min_power - expected_min) < 1e-6, f"Min mismatch at slot {t}"
        assert abs(agg_profile[t].max_power - expected_max) < 1e-6, f"Max mismatch at slot {t}"

    # Check declared bounds are feasible
    total_energy_min = sum(ts.min_power for ts in agg_profile)
    total_energy_max = sum(ts.max_power for ts in agg_profile)

    assert aggregated.get_min_overall_alloc() <= total_energy_max + 1e-6
    assert aggregated.get_max_overall_alloc() >= total_energy_min - 1e-6
    
    for fo in [fo1, fo2, fo3]:
        fo.print_flexoffer()
    aggregated.print_flexoffer()