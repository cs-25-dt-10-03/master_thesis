import pandas as pd
from classes.electricVehicle import ElectricVehicle
import numpy as np
from multiprocessing import Pool
from config import config
from datetime import datetime, timedelta
from flexoffer_logic import set_time_resolution
from database.parser import fo_parser

def simulate_fleet(
    num_evs: int = config.NUM_EVS,
    start_date = config.SIMULATION_START_DATE,
    simulation_days: int = config.SIMULATION_DAYS
):
    """
    Simulates a fleet of EVs over multiple days and generates FlexOffers.
      - num_evs: number of EVs (from config)
      - start_date: date to begin simulation
      - simulation_days: how many days to run
    Returns: (list_of_flexoffers, list_of_dfos)
    """

    if config.USE_SYNTHETIC:
        fleet = [
            ElectricVehicle(
                i,
                capacity=np.random.normal(60, 10),
                charging_power=np.random.choice([7.2, 11, 22]),
                soc_min=0.7,
                soc_max=0.9,
                charging_efficiency=1
            )
            for i in range(num_evs)
        ]

        # 3) Prepare all (EV, day) tasks
        days = pd.date_range(start=start_date, periods=simulation_days, freq=pd.DateOffset(days=1))
        fos = []
        dfos = []


        for day in days:
            for ev in fleet:
                prof = ev.sample_day_profile(day)
                if prof is None:
                    continue
                arrival, departure, soc = prof
                fo  = ev.create_synthetic_flex_offer(arrival, departure, soc)
                dfo = ev.create_dfo(arrival, departure - arrival, 4)
                if fo is None or dfo is None:
                    continue
                fos.append(fo)
                dfos.append(dfo)
        return fos, dfos
    
    else:
        # Non-synthetic (historical) mode
        start_date = pd.to_datetime(config.SIMULATION_START_DATE)
        stop_date  = start_date + timedelta(config.SIMULATION_DAYS)

        # Convert to slot indices for fo_parser
        origin = datetime(2020, 1, 1, 0, 0)
        start = (start_date - origin).total_seconds() / config.TIME_RESOLUTION
        stop  = (stop_date  - origin).total_seconds() / config.TIME_RESOLUTION

        fos = fo_parser(start, stop)
        return fos, []
