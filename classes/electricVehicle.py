from datetime import datetime, timedelta
from typing import Optional
from classes.flexOffer import flexOffer

class ElectricVehicle:
    def __init__(self, vehicle_id: str, 
                 capacity: float,
                 soc_min: float,
                 soc_max: float,
                 charging_power: float,
                 charging_efficiency: float,
                 initial_soc: Optional[float] = None):
        
        self.vehicle_id = vehicle_id
        self.capacity = capacity
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.charging_power = charging_power
        self.charging_efficiency = charging_efficiency
        self.current_soc = initial_soc if initial_soc is not None else soc_min

    def create_flex_offer(self,
                          charging_window_start: datetime,
                          charging_window_end: datetime,
                          time_slot_resolution: timedelta,
                          required_duration: timedelta) -> flexOffer:

        valid_resolutions = {timedelta(hours=1), timedelta(minutes=15)}

        if time_slot_resolution not in valid_resolutions:
            raise ValueError("Time slot resolution must be either 1 hour or 15 minutes.")
        
        if charging_window_end - charging_window_start < required_duration:
            raise ValueError("Charging window is too short for the required charging duration.")
        
        latest_start = charging_window_end - required_duration

        flex_duration = required_duration
        num_slots = int(flex_duration / time_slot_resolution)
        energy_per_slot = self.charging_power * (time_slot_resolution.total_seconds() / 3600)

        energy_profile = [energy_per_slot * self.charging_efficiency for _ in range(num_slots)]
        
        min_energy = self.soc_min * self.capacity
        max_energy = self.soc_max * self.capacity
        
        flex_offer = flexOffer(
            offer_id=self.vehicle_id,
            earliest_start=charging_window_start,
            latest_start=latest_start,
            duration=flex_duration,
            energy_profile=energy_profile,
            min_energy=min_energy,
            max_energy=max_energy,
            total_energy_limit=self.capacity
        )
        return flex_offer

    def __repr__(self):
        return (f"<EV {self.vehicle_id}: SoC={self.current_soc*100:.0f}% "
            f"of {self.capacity} kWh>")
