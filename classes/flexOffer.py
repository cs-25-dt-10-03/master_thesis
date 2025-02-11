from datetime import datetime, timedelta
from typing import Optional, List

class flexOffer:
    def __init__(self, offer_id: str,
                 earliest_start: datetime,
                 latest_start: datetime,
                 duration: timedelta,
                 energy_profile: List[float],
                 min_energy: Optional[float] = None,
                 max_energy: Optional[float] = None,
                 total_energy_limit: Optional[float] = None):
        
        self.offer_id = offer_id
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        self.duration = duration
        self.energy_profile = energy_profile
        self.min_energy = min_energy
        self.max_energy = max_energy
        self.total_energy_limit = total_energy_limit

    @property
    def total_energy(self) -> float:
        return sum(self.energy_profile)
    
    @property
    def get_earliest(self):
        return self.earliest_start.replace(minute=0, second=0, microsecond=0)
    
    @property
    def get_latest(self):
        return self.latest_start.replace(minute=0, second=0, microsecond=0)
    
    def __repr__(self):
        return (f"<FlexOffer id={self.offer_id} "
                f"start_window=({self.get_earliest} - {self.get_latest}) "
                f"duration={self.duration} total_energy={self.total_energy}>")