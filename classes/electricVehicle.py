from datetime import datetime, timedelta
from typing import Tuple
import numpy as np
import pandas as pd
from helpers import dt_to_unix
from scipy.stats import lognorm, beta
from flexoffer_logic import Flexoffer, TimeSlice, DFO
from config import config


class ElectricVehicle:
    def __init__(self, vehicle_id: int,
                 capacity: float,
                 soc_min: float,
                 soc_max: float,
                 charging_power: float,
                 charging_efficiency: float,
                 ):

        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charging_power = charging_power
        self.charging_efficiency = charging_efficiency
        self.current_soc = self.sample_soc()

    def sample_soc(self) -> float:
        alpha, beta_param = 2, 5
        sampled_soc = beta.rvs(alpha, beta_param)
        return 0.2 + (self.soc_max - self.soc_min) * sampled_soc

    def sample_start_times(self) -> Tuple[datetime, datetime]:
        arrival_mu = np.log(18)
        arrival_sigma = 0.1

        arrival_hour = int(lognorm.rvs(s=arrival_sigma, scale=np.exp(arrival_mu)))
        charging_window_start = datetime.now().replace(year=2024, hour=arrival_hour, minute=0, second=0, microsecond=0)

        dep_mu = np.log(8)
        dep_sigma = 0.1

        departure_hour = int(lognorm.rvs(s=dep_sigma, scale=np.exp(dep_mu)))
        charging_window_end = datetime.now().replace(year=2024, hour=departure_hour, minute=0, second=0, microsecond=0)

        if departure_hour < arrival_hour:
            charging_window_end += timedelta(days=1)

        return charging_window_start, charging_window_end

    def create_flexoffer(data: pd.DataFrame) -> Flexoffer:
        if len(data) > 24:
            raise Exception("Alt for mange elementer:", len(data)) 

    def create_synthetic_flex_offer(self, tec_fo: bool) -> Flexoffer:
        earliest_start, end_time = self.sample_start_times()

        # Determine required charging energy
        required_energy = (self.soc_max - self.current_soc) * self.capacity if tec_fo else 0

        # Compute necessary charging duration
        if required_energy > 0:
            charging_time_hours = required_energy / (self.charging_power * self.charging_efficiency)
        else:
            charging_time_hours = 0

        time_slot_resolution = config.TIME_RESOLUTION / 3600
        charging_time_hours = round(charging_time_hours / time_slot_resolution) * time_slot_resolution

        # Compute the latest possible start time (ensuring it’s aligned to the time resolution)
        latest_start = end_time - timedelta(hours=charging_time_hours)
        latest_start = latest_start.replace(second=0, microsecond=0)  # Remove seconds/milliseconds
        remainder = latest_start.minute % (config.TIME_RESOLUTION // 60)
        latest_start -= timedelta(minutes=remainder)  # Snap to valid resolution step

        # Generate time slices for charging profile
        duration = int((end_time - earliest_start).total_seconds() // config.TIME_RESOLUTION)
        max_energy_per_slot = self.charging_power * (config.TIME_RESOLUTION / 3600) * self.charging_efficiency
        energy_profile = [TimeSlice(0.0, max_energy_per_slot) for _ in range(duration)]

        # Define min and max overall allocation
        min_energy = self.soc_min * self.capacity if tec_fo else 0
        max_energy = self.soc_max * self.capacity if tec_fo else 0

        print(f"min energy: {min_energy}, max energy: {max_energy}")

        # Debugging checks
        assert latest_start < end_time, f"Error: latest_start ({latest_start}) should not equal end_time ({end_time})!"
        assert required_energy >= 0, f"Error: required_energy is negative ({required_energy})!"
        assert charging_time_hours >= time_slot_resolution, f"Error: charging_time ({charging_time_hours} h) is too small!"

        return Flexoffer(
            offer_id=self.vehicle_id,
            earliest_start=dt_to_unix(earliest_start),
            latest_start=dt_to_unix(latest_start),
            end_time=dt_to_unix(end_time),
            profile=energy_profile,
            duration=duration,
            min_overall_alloc=min_energy,
            max_overall_alloc=max_energy
        )

    def create_dfo(self, charging_window_start: datetime, duration, numsamples) -> DFO:

        time_slot_resolution = timedelta(seconds=config.TIME_RESOLUTION)

        num_slots = int(duration / time_slot_resolution) + 1

        initial_energy = self.current_soc * self.capacity
        target_min_energy = self.soc_min * self.capacity
        target_max_energy = self.soc_max * self.capacity

        additional_min = max(target_min_energy - initial_energy, 0)
        additional_max = max(target_max_energy - initial_energy, 0)
        min_prev = []
        max_prev = []

        for i in range(num_slots):
            min_prev.append(max(additional_min - self.charging_power * i, 0))
            max_prev.append(min(self.charging_power * i, additional_max))
        min_prev.reverse()

        # Convert `charging_window_start` to integer Unix timestamp
        earliest_start_timestamp = int(charging_window_start.timestamp())

        dfo = DFO(self.vehicle_id, min_prev, max_prev, numsamples, self.charging_power, additional_min, additional_max, earliest_start_timestamp)
        dfo.generate_dependency_polygons()
        return dfo

    def update_soc(self, charged_energy):
        new_energy = self.capacity * self.current_soc + charged_energy
        self.current_soc = new_energy / self.capacity

    def __repr__(self):
        return (f"<EV {self.vehicle_id}: SoC={self.current_soc*100:.0f}% "
                f"of {self.capacity} kWh>")
