from datetime import datetime, timedelta
import math
import pandas as pd
from typing import Tuple
import numpy as np
from helpers import dt_to_unix, round_datetime_to_resolution
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
                 soc=None
                 ):

        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charging_power = charging_power
        self.charging_efficiency = charging_efficiency
        if soc is None:
            self.current_soc = self.sample_soc()
        else:
            self.current_soc = soc

    def sample_soc(self) -> float:
        alpha, beta_param = 2, 5
        sampled_soc = beta.rvs(alpha, beta_param)
        return 0.2 + (self.soc_max - self.soc_min) * sampled_soc

    def sample_day_profile(self, day_start):
        """
        Samples whether EV is active today and if yes, generates arrival/departure/SoC.
        day_start: datetime at 00:00 of the day.
        Now constrains window to 06:00–18:00 on the same calendar day.
        """
        # 80% chance weekday, 50% chance weekend
        is_weekend   = day_start.weekday() >= 5
        active_prob  = 0.8 if not is_weekend else 0.5
        # if np.random.rand() > active_prob:
        #     return None  # EV is inactive today

        # --- ARRIVAL between 10:00 and 12:00 ---
        arrival_hour = np.random.uniform(12, 14)
        arrival = day_start + timedelta(hours=arrival_hour)

        # --- DEPARTURE between (arrival+1h) and 18:00 ---
        earliest_dep = arrival_hour + 1
        latest_dep   = 16
        # if someone arrives at 17, earliest_dep=16, so departure_hour==16
        if earliest_dep >= latest_dep:
            departure_hour = latest_dep
        else:
            departure_hour = np.random.uniform(earliest_dep, latest_dep)
        departure = day_start + timedelta(hours=departure_hour)

        # State of charge at arrival (beta distribution)
        arrival_soc = np.random.beta(2, 5) * 0.8  # between 0 and 0.4
        arrival_soc = np.clip(arrival_soc, 0.1, 0.4)

        return arrival, departure, arrival_soc


    def create_synthetic_flex_offer(self, arrival, departure, arrival_soc, target_soc=0.9, resolution_seconds=None):
        """
        Creates a FlexOffer from a sampled day profile with time rounding.
        """
        if resolution_seconds is None:
            resolution_seconds = config.TIME_RESOLUTION

        # Round arrival and departure
        arrival_rounded = round_datetime_to_resolution(arrival, resolution_seconds, "down")
        departure_rounded = round_datetime_to_resolution(departure, resolution_seconds, "up")

        # Charging need
        needed_energy = (target_soc - arrival_soc) * self.capacity
        if needed_energy <= 0:
            return None  # No charging needed

        charging_power = self.charging_power * self.charging_efficiency

        duration = int((departure_rounded - arrival_rounded).total_seconds() // resolution_seconds)

        if duration <= 0:
            return None

        max_energy_per_slot = charging_power * (resolution_seconds / 3600)
        max_possible_energy = duration * max_energy_per_slot

        energy_min = max_possible_energy * 0.90
        energy_max = max_possible_energy * 1.1

        profile = [TimeSlice(0, charging_power) for _ in range(duration)]

        return Flexoffer(
            offer_id=self.vehicle_id,
            earliest_start=dt_to_unix(arrival_rounded),
            latest_start=max(dt_to_unix(arrival_rounded), dt_to_unix(departure_rounded) - duration * resolution_seconds),
            end_time=dt_to_unix(departure_rounded),
            profile=profile,
            duration=duration,
            min_overall_alloc=energy_min,
            max_overall_alloc=energy_max
        )

    def sample_start_times(self) -> Tuple[datetime, datetime]:
        arrival_mu = np.log(18)
        arrival_sigma = 0.1

        dep_mu = np.log(8)
        dep_sigma = 0.1

        arrival_hour = int(lognorm.rvs(s=arrival_sigma, scale=np.exp(arrival_mu)))
        if arrival_hour >= 24:
            arrival_hour = 23

        departure_hour = int(lognorm.rvs(s=dep_sigma, scale=np.exp(dep_mu)))

        start_day = pd.to_datetime(config.SIMULATION_START_DATE)

        charging_window_start = start_day.replace(hour=arrival_hour, minute=0, second=0, microsecond=0)
        charging_window_end = start_day.replace(hour=departure_hour, minute=0, second=0, microsecond=0)

        if departure_hour < arrival_hour:
            charging_window_end += timedelta(days=1)

            return charging_window_start, charging_window_end

    def create_flexoffer(self, start, stop) -> Flexoffer:
        earliest_start = datetime(2020, 1, 1, 0, 0, 0) + timedelta(hours=start)
        end_time = datetime(2020, 1, 1, 0, 0, 0) + timedelta(hours=stop)

        # Determine required charging energy
        required_energy = (self.soc_max - self.current_soc) * self.capacity
        if required_energy < 0 or math.isnan(required_energy):
            required_energy = 0

        # Compute necessary charging duration
        if required_energy > 0:
            charging_time_hours = required_energy / (self.charging_power * self.charging_efficiency)
        else:
            charging_time_hours = 0

        time_slot_resolution = config.TIME_RESOLUTION / 3600
        charging_time_hours = round(charging_time_hours / time_slot_resolution) * time_slot_resolution
        if charging_time_hours == 0:
            charging_time_hours = 1

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
        min_energy = self.current_soc * self.capacity
        max_energy = self.soc_max * self.capacity

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

    # def create_synthetic_flex_offer(self, tec_fo: bool) -> Flexoffer:
    #     earliest_start, end_time = self.sample_start_times()

    #     # Determine required charging energy
    #     required_energy = (self.soc_max - self.current_soc) * self.capacity if tec_fo else 0

    #     # Compute necessary charging duration
    #     if required_energy > 0:
    #         charging_time_hours = required_energy / (self.charging_power * self.charging_efficiency)
    #     else:
    #         charging_time_hours = 0

    #     time_slot_resolution = config.TIME_RESOLUTION / 3600
    #     charging_time_hours = round(charging_time_hours / time_slot_resolution) * time_slot_resolution

    #     # Compute the latest possible start time (ensuring it’s aligned to the time resolution)
    #     latest_start = end_time - timedelta(hours=charging_time_hours)
    #     latest_start = latest_start.replace(second=0, microsecond=0)  # Remove seconds/milliseconds
    #     remainder = latest_start.minute % (config.TIME_RESOLUTION // 60)
    #     latest_start -= timedelta(minutes=remainder)  # Snap to valid resolution step

    #     # Generate time slices for charging profile
    #     duration = int((end_time - earliest_start).total_seconds() // config.TIME_RESOLUTION)
    #     max_energy_per_slot = self.charging_power * (config.TIME_RESOLUTION / 3600) * self.charging_efficiency
    #     energy_profile = [TimeSlice(0.0, max_energy_per_slot) for _ in range(duration)]

    #     # Define min and max overall allocation
    #     min_energy = self.soc_min * self.capacity if tec_fo else 0
    #     max_energy = self.soc_max * self.capacity if tec_fo else 0

    #     # Debugging checks
    #     assert latest_start < end_time, f"Error: latest_start ({latest_start}) should not equal end_time ({end_time})!"
    #     assert required_energy >= 0, f"Error: required_energy is negative ({required_energy})!"
    #     assert charging_time_hours >= time_slot_resolution, f"Error: charging_time ({charging_time_hours} h) is too small!"

    #     return Flexoffer(
    #         offer_id=self.vehicle_id,
    #         earliest_start=dt_to_unix(earliest_start),
    #         latest_start=dt_to_unix(latest_start),
    #         end_time=dt_to_unix(end_time),
    #         profile=energy_profile,
    #         duration=duration,
    #         min_overall_alloc=min_energy,
    #         max_overall_alloc=max_energy
    #     )

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
        #dfo.calculate_latest_start_time()
        return dfo

    def update_soc(self, charged_energy):
        new_energy = self.capacity * self.current_soc + charged_energy
        self.current_soc = new_energy / self.capacity

    def __repr__(self):
        return (f"<EV {self.vehicle_id}: SoC={self.current_soc*100:.0f}% "
                f"of {self.capacity} kWh>")