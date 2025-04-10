import pytest
from datetime import datetime, timedelta
from flexoffer_logic import Flexoffer, TimeSlice
from classes.electricVehicle import ElectricVehicle
from datetime import datetime, timedelta
from config import config


@pytest.fixture
def ev():
    return ElectricVehicle(
        vehicle_id=1,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7.0,
        charging_efficiency=0.95
    )

def test_create_flex_offer(ev):
    fo = ev.create_synthetic_flex_offer(tec_fo=True)
    assert fo.get_est() <= fo.get_lst()
    assert fo.get_lst() < fo.get_et()
    assert len(fo.get_profile()) > 0

def test_sample_soc_within_bounds(ev):
    soc = ev.sample_soc()
    assert ev.soc_min <= soc <= ev.soc_max

def test_update_soc(ev):
    initial_soc = ev.current_soc
    ev.update_soc(10)
    assert ev.current_soc > initial_soc




# def test_flex_offer_possible_start_times():
#     t0 = datetime(2024, 1, 1, 8, 0)
#     t1 = datetime(2024, 1, 1, 10, 0)
#     duration = t1 - t0
#     energy_profile = [(0, 7) for _ in range(2)]
#     fo = Flexoffer(
#         offer_id=1,
#         earliest_start=t0,
#         latest_start=t1,
#         end_time=t1,
#         duration=duration,
#         energy_profile=energy_profile
#     )
#     possible_times = fo.possible_start_times(time_resolution_minutes=config.TIME_RESOLUTION)
#     assert possible_times[0] == t0
#     assert possible_times[-1] <= t1
