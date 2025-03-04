import pytest
from datetime import datetime, timedelta
from classes.flexOffer import flexOffer
from classes.electricVehicle import ElectricVehicle

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
    flex_offer = ev.create_flex_offer(tec_fo=True)
    assert flex_offer.earliest_start <= flex_offer.latest_start
    assert flex_offer.latest_start < flex_offer.end_time
    assert len(flex_offer.energy_profile) > 0

def test_sample_soc_within_bounds(ev):
    soc = ev.sample_soc()
    assert ev.soc_min <= soc <= ev.soc_max

def test_update_soc(ev):
    initial_soc = ev.current_soc
    ev.update_soc(10)
    assert ev.current_soc > initial_soc