import pandas as pd
from classes.electricVehicle import ElectricVehicle
import numpy as np
from config import config

def simulate_fleet(num_evs=config.NUM_EVS, start_date=config.SIMULATION_START_DATE, simulation_days=config.SIMULATION_DAYS):
    """
    Simulates a fleet of EVs over multiple days and generates FlexOffers.
    num_evs: number of evs given in config.py file
    start_date: start date for simulatio
    simulation_days: number of days
    """
    fleet = []
    for i in range(num_evs):
        ev = ElectricVehicle(i, 
                            capacity=np.random.normal(60, 10), 
                            charging_power=np.random.choice([7.2,11,22]),
                            soc_min=0.7,
                            soc_max=0.9,
                            charging_efficiency = 1)
        fleet.append(ev)

    fos = []
    dfos = []

    for day in pd.date_range(start=start_date, periods=simulation_days, freq='D'):
        for ev in fleet:
            profile = ev.sample_day_profile(day)
            if profile:
                arrival, departure, soc = profile
                fo = ev.create_flex_offer(arrival, departure, soc)
                if fo:
                    fos.append(fo)
                duration = departure - arrival
                dfo = ev.create_dfo(arrival, duration, 4)
                if dfo:
                    dfos.append(dfo)
    return fos, dfos
