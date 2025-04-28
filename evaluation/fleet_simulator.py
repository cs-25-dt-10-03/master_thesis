import pandas as pd
from classes.electricVehicle import ElectricVehicle
import numpy as np

def simulate_fleet(num_evs, start_date, simulation_days):
    """
    Simulates a fleet of EVs over multiple days and generates FlexOffers.
    num_evs: number of evs given in config.py file
    start_date: start date for simulation given in config.py file
    simulation_days: number of days given in the config.py file
    """
    start_date = pd.to_datetime(start_date)
    fleet = []
    for i in range(num_evs):
        ElectricVehicle(i, capacity_kWh=np.random.normal(60, 10),
                          charging_power_kW=np.random.choice([7.2,11,22]))
    

    offers = []
    for day in pd.date_range(start=start_date, periods=simulation_days, freq='D'):
        for ev in fleet:
            profile = ev.sample_day_profile(day)
            if profile:
                arrival, departure, soc = profile
                fo = ev.create_flex_offer(arrival, departure, soc)
                if fo:
                    offers.append(fo)
    return offers
