import datetime
import math
import pandas as pd
from typing import Tuple, Optional
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
        # Uniform distribution around mean 40%, per [17]
        return np.random.uniform(0.3, 0.5)

    def sample_day_profile(self, day_start: datetime, resolution_seconds: Optional[int] = None):
        """
        Samples EV plug-in and plug-out times using log-normal distributions,
        matching the approach in the reference papers. The charging window
        is taken as between 17:00 on the given calendar day and 08:00 the next day.
        Returns None if EV inactive, otherwise (arrival, departure, soc).
        """
        if resolution_seconds is None:
            resolution_seconds = config.TIME_RESOLUTION

        # ─────────────────────────────────────────────────────────
        # force sampling window to the calendar‐day at midnight
        calendar_day = day_start.replace(hour=0, minute=0, second=0, microsecond=0)

        # Determine if EV is used today (weekdays/weekends probabilities)
        is_weekend   = calendar_day.weekday() >= 5
        active_prob  = 0.80 if not is_weekend else 0.20
        if np.random.rand() > active_prob:
            return None

        # Log-normal parameters tuned from reference [1]
        arrival_mu    = np.log(18.25)  # mean ~18:15
        arrival_sigma = 0.22           # narrower spread for sharp arrival peak
        depart_mu     = np.log(31.0)   # mean ~07:00 next day
        depart_sigma  = 0.18           # tighter spread for departure

        # Sample arrival between 17:00 and 22:00 on calendar_day
        while True:
            sample_h = lognorm.rvs(s=arrival_sigma, scale=np.exp(arrival_mu))
            if 17 <= sample_h < 22:
                arrival = calendar_day + datetime.timedelta(hours=sample_h)
                break

        # Sample plug-out between 00:00 and 08:00 next calendar day
        next_midnight = (calendar_day + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        while True:
            sample_h = lognorm.rvs(s=depart_sigma, scale=np.exp(depart_mu))
            # shift to next day hours by subtracting 24
            if 26 <= sample_h < 32:
                depart_hour = sample_h - 24
                if 0 <= depart_hour <= 8:
                    departure = next_midnight + datetime.timedelta(hours=depart_hour)
                    break

        
        # Round to time resolution
        arrival_rounded   = round_datetime_to_resolution(arrival,            resolution_seconds, "down")
        departure_rounded = round_datetime_to_resolution(departure, resolution_seconds, "down")
        # Sample state of charge at plug-in uniformly [17%, 43%]
        arrival_soc = np.random.uniform(0.17, 0.43)

        return arrival_rounded, departure_rounded, arrival_soc


    def create_synthetic_flex_offer(self, arrival, departure, arrival_soc, target_soc=0.9, resolution_seconds=None):
        if resolution_seconds is None:
            resolution_seconds = config.TIME_RESOLUTION

        dt_hours = resolution_seconds / 3600.0
        power = self.charging_power * self.charging_efficiency
        energy_per_slot = power * dt_hours

        est = dt_to_unix(arrival)
        et  = dt_to_unix(departure)

        # Total time available (slots)
        total_slots = int((et - est) // resolution_seconds)
        if total_slots <= 0:
            return None

        # Energy need
        needed_energy = (target_soc - arrival_soc) * self.capacity
        if needed_energy <= 0:
            return None

        # Minimum slots needed to meet energy need at full power
        min_required_slots = math.ceil(needed_energy / energy_per_slot)
        if min_required_slots > total_slots:
            return None

        # Compute latest start time such that min_required_slots still fit before et
        lst = et - min_required_slots * resolution_seconds

        # Build FO
        # max power (kW) in each slice:
        profile = [TimeSlice(0.0, power) for _ in range(min_required_slots)]
        min_alloc = needed_energy
        max_alloc = min_alloc * 1.1

        return Flexoffer(
            offer_id=self.vehicle_id,
            earliest_start=est,
            latest_start=lst,
            end_time=et,
            profile=profile,
            duration=min_required_slots,
            min_overall_alloc=min_alloc,
            max_overall_alloc=max_alloc
        )

    def create_dfo(self, charging_window_start: datetime, duration, numsamples) -> DFO:

        time_slot_resolution = datetime.timedelta(seconds=config.TIME_RESOLUTION)

        num_slots = int(duration // time_slot_resolution) + 1

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
    


    # -- This is for the actual data from SDU, keeping it here for now -- # 

    # def create_flexoffer(self, start, stop) -> Flexoffer:
    #     earliest_start = datetime(2020, 1, 1, 0, 0, 0) + datetime.timedelta(hours=start)
    #     end_time = datetime(2020, 1, 1, 0, 0, 0) + datetime.timedelta(hours=stop)

    #     # Determine required charging energy
    #     required_energy = (self.soc_max - self.current_soc) * self.capacity
    #     if required_energy < 0 or math.isnan(required_energy):
    #         required_energy = 0

    #     # Compute necessary charging duration
    #     if required_energy > 0:
    #         charging_time_hours = required_energy / (self.charging_power * self.charging_efficiency)
    #     else:
    #         charging_time_hours = 0

    #     time_slot_resolution = config.TIME_RESOLUTION / 3600
    #     charging_time_hours = round(charging_time_hours / time_slot_resolution) * time_slot_resolution
    #     if charging_time_hours == 0:
    #         charging_time_hours = 1

    #     # Compute the latest possible start time (ensuring it’s aligned to the time resolution)
    #     latest_start = end_time - datetime.timedelta(hours=charging_time_hours)
    #     latest_start = latest_start.replace(second=0, microsecond=0)  # Remove seconds/milliseconds
    #     remainder = latest_start.minute % (config.TIME_RESOLUTION // 60)
    #     latest_start -= datetime.timedelta(minutes=remainder)  # Snap to valid resolution step

    #     # Generate time slices for charging profile
    #     duration = int((end_time - earliest_start).total_seconds() // config.TIME_RESOLUTION)
    #     max_energy_per_slot = self.charging_power * (config.TIME_RESOLUTION / 3600) * self.charging_efficiency
    #     energy_profile = [TimeSlice(0.0, max_energy_per_slot) for _ in range(duration)]

    #     # Define min and max overall allocation
    #     min_energy = self.current_soc * self.capacity
    #     max_energy = self.soc_max * self.capacity

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
