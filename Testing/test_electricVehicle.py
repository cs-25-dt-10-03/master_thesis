import pytest
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import beta, lognorm
from classes.electricVehicle import ElectricVehicle
from classes.flexOffer import FlexOffer
from classes.DFO import DFO
from config import config

def test_electric_vehicle_sample_soc():
    ev = ElectricVehicle(
        vehicle_id=1,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7,
        charging_efficiency=0.95
    )
    soc = ev.current_soc
    assert ev.soc_min <= soc <= ev.soc_max

def test_electric_vehicle_sample_start_times():
    ev = ElectricVehicle(
        vehicle_id=2,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7,
        charging_efficiency=0.95
    )
    start_time, end_time = ev.sample_start_times()
    # The arrival (start) time should be before the departure (end) time.
    assert start_time < end_time

def test_create_flex_offer():
    ev = ElectricVehicle(
        vehicle_id=3,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7,
        charging_efficiency=0.95
    )
    # Request a technical flex offer so that min_energy, etc. are set.
    fo = ev.create_flex_offer(tec_fo=True)
    # Check that we obtain a flexOffer instance.
    assert isinstance(fo, FlexOffer)
    # The energy profile should be non-empty.
    assert len(fo.energy_profile) > 0
    # For tec flex offers, min_energy should be defined.
    assert fo.min_energy is not None

def test_create_dfo():
    ev = ElectricVehicle(
        vehicle_id=4,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7,
        charging_efficiency=0.95
    )
    start_time, end_time = ev.sample_start_times()
    duration = end_time - start_time
    dfo = ev.create_dfo(start_time, end_time, duration, numsamples=5)
    # After generating dependency polygons, dfo.polygons should not be empty.
    assert len(dfo.polygons) > 0

def test_update_soc():
    ev = ElectricVehicle(
        vehicle_id=5,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7,
        charging_efficiency=0.95
    )
    initial_soc = ev.current_soc
    ev.update_soc(10)  # Charge with 10 kWh
    # The state of charge should increase.
    assert ev.current_soc > initial_soc
