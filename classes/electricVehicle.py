from datetime import datetime, timedelta
from typing import Optional, Tuple
import numpy as np
from scipy.stats import lognorm, beta
from classes.flexOffer import FlexOffer
from config import config
from classes.DFO import DFO

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
        return self.soc_min + (self.soc_max - self.soc_min) * sampled_soc

    def sample_start_times(self) -> Tuple[datetime, datetime]:
        arrival_mu = np.log(18)
        arrival_sigma = 0.1 

        arrival_hour = int(lognorm.rvs(s=arrival_sigma, scale=np.exp(arrival_mu)))
        charging_window_start= datetime.now().replace(year=2024, hour=arrival_hour, minute=0, second=0, microsecond=0)

        dep_mu = np.log(8)
        dep_sigma = 0.1

        departure_hour = int(lognorm.rvs(s=dep_sigma, scale=np.exp(dep_mu)))
        charging_window_end = datetime.now().replace(year=2024, hour=departure_hour, minute=0, second=0, microsecond=0)
        
        if departure_hour < arrival_hour:
            charging_window_end += timedelta(days=1)


        print(f"Arrival Time: {charging_window_start.strftime('%H:%M')}")
        print(f"Departure Time: {charging_window_end.strftime('%H:%M')}")


        return charging_window_start, charging_window_end


    def create_flex_offer(self, tec_fo: bool = False) -> FlexOffer:
        earliest_start, end_time = self.sample_start_times()

        if tec_fo == True:
            target_soc = self.soc_max  #The tec fo should have the capability to reach max soc.
            required_energy = (target_soc - self.current_soc) * self.capacity  # kWh
        else:
            required_energy = 0

        # **Compute Charging Time Needed**
        if required_energy > 0:
            charging_time = required_energy / (self.charging_power * self.charging_efficiency)  # Hours
            charging_time = timedelta(minutes=charging_time, seconds=0, milliseconds=0)
        else:
            charging_time = timedelta(minutes=0)

        latest_start = end_time - charging_time
        duration = end_time - latest_start
        time_slot_resolution = timedelta(seconds = config.TIME_RESOLUTION)
        num_slots = int((end_time - earliest_start) / time_slot_resolution) 

        max_energy_per_slot = self.charging_power * (time_slot_resolution.total_seconds() / config.TIME_RESOLUTION) * self.charging_efficiency
        # (min, max) tuple format
        energy_profile = [(float(0), max_energy_per_slot) for _ in range(num_slots)]
        
        if tec_fo:
            min_energy = self.soc_min * self.capacity
            max_energy = self.soc_max * self.capacity
            total_energy_limit = self.capacity
        else:
            min_energy = None
            max_energy = None
            total_energy_limit = None
        
        flex_offer = FlexOffer(
            offer_id=self.vehicle_id,
            earliest_start=earliest_start,
            latest_start=latest_start,
            end_time=end_time,
            duration=duration,
            energy_profile=energy_profile,
            min_energy=min_energy,
            max_energy=max_energy,
            total_energy_limit=total_energy_limit
        )
        return flex_offer
    
    def create_dfo(self, charging_window_start: datetime, charging_window_end: datetime, duration, numsamples) -> DFO:

        time_slot_resolution = timedelta(seconds = config.TIME_RESOLUTION)     

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
        dfo = DFO(self.vehicle_id, min_prev, max_prev, numsamples, self.charging_power, additional_min, additional_max, charging_window_start)
        dfo.generate_dependency_polygons()
        return dfo

    def update_soc(self, charged_energy):
        new_energy = self.capacity * self.current_soc + charged_energy
        self.current_soc = new_energy / self.capacity

    def __repr__(self):
        return (f"<EV {self.vehicle_id}: SoC={self.current_soc*100:.0f}% "
            f"of {self.capacity} kWh>")
