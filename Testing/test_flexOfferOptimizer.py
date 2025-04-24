from optimization.flexOfferOptimizer import optimize, mFRR_reserve_only, optimize_Spot_and_mfrr, sequential_schedule_mfrr_then_spot, optimize_spot_reserve_activation
from classes.electricVehicle import ElectricVehicle
from database.dataManager import get_prices_in_range
import pulp
from datetime import datetime
import pytest
from config import config
import pandas as pd
from flexoffer_logic import Flexoffer, TimeSlice

def test_optimizer():
    
    ev = ElectricVehicle(
        vehicle_id=1,
        capacity=100,
        soc_min=0.7,
        soc_max=0.9,
        charging_power=7.0,
        charging_efficiency=0.95
    )


    fo = ev.create_synthetic_flex_offer(tec_fo=True)
    fos = []
    fos.append(fo)

