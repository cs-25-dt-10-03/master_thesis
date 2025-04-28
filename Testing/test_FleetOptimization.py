
from optimization.flexOfferOptimizer import optimize
from classes.electricVehicle import ElectricVehicle
from database.dataManager import get_prices_in_range
import pulp
from datetime import datetime
import pytest
from config import config
import pandas as pd
from flexoffer_logic import Flexoffer, TimeSlice
from optimization.flexOfferOptimizer import optimize
from database.dataManager import load_and_prepare_prices
import pandas as pd
from config import config
import os



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

solution = optimize(fos)
