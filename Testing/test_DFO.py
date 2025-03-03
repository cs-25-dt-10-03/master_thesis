import pytest
from datetime import datetime, timedelta
from classes.flexOffer import flexOffer
from classes.DFO import DFO
from classes.electricVehicle import ElectricVehicle
from aggregation.DFO_aggregation import agg2to1

@pytest.fixture
def ev1():
    return ElectricVehicle(
        vehicle_id="TeslaModelY_1",
        capacity=75.0,
        soc_min=0.20,
        soc_max=0.80,
        charging_power=7.0,
        charging_efficiency=0.84,
    )

@pytest.fixture
def ev2():
    return ElectricVehicle(
        vehicle_id="TeslaModelS_1",
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
    )


def test_create_dfos(ev1, ev2):
    charging_window_start = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)
    charging_window_end = charging_window_start + timedelta(hours=8)
    duration = timedelta(hours=3) 

    dfo1 = ev1.create_dfo(charging_window_start,
                                   charging_window_end,
                                   duration,
                                   numsamples=4)

    dfo2 = ev2.create_dfo(charging_window_start+timedelta(hours=1),
                                   charging_window_end,
                                   duration,
                                   numsamples=4)
    dfo3 = agg2to1(dfo1, dfo2, 4)