from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class flexOffer:
    def __init__(self, offer_id: str,
                 earliest_start: datetime,
                 latest_start: datetime,
                 energy_profile: List[Tuple[float, float]],
                 min_energy: Optional[float] = None,
                 max_energy: Optional[float] = None,
                 total_energy_limit: Optional[float] = None):
        
        self.offer_id = offer_id
        self.earliest_start = earliest_start
        self.latest_start = latest_start
        self.energy_profile = energy_profile
        self.min_energy = min_energy
        self.scheduled_start = None
        self.scheduled_energy_profile = None
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
    
    def plot(self, schedule_start: Optional[datetime] = None, show_window: bool = True):
        if schedule_start is None:
            schedule_start = self.earliest_start

        end_time = self.latest_start + self.duration

        num_slots = len(self.energy_profile)
        time_step: timedelta = self.duration / num_slots

        times = []
        for i in range(num_slots):
            times.append(schedule_start + i * time_step)
        times_num = [mdates.date2num(t) for t in times]

        min_values = [e[0] for e in self.energy_profile]
        flexible_values = [e[1] - e[0] for e in self.energy_profile]
 
        fig, ax = plt.subplots(figsize=(7, 3))
 
        #first plot minimum as blue, then flexible as orange
        ax.bar(times_num, min_values, width=1/24, align='edge',color='blue', edgecolor='black', label='Minimum Energy')
        ax.bar(times_num, flexible_values, width=1/24, bottom=min_values, align='edge', color='orange', edgecolor='black', label='Flexible Energy')
 
        if show_window:
            window_left = mdates.date2num(self.earliest_start)
            window_right = mdates.date2num(self.latest_start)
            ax.axvspan(window_left, window_right, color='gray', alpha=0.2)
 
        ax.set_title(f"FlexOffer {self.offer_id}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Energy (kWh)")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        fig.autofmt_xdate()
 
        # Plot key time markers
        t_es = mdates.date2num(self.earliest_start)
        t_ls = mdates.date2num(self.latest_start)
        t_le = mdates.date2num(end_time)
        ax.axvline(x=t_es, color='black', linestyle='-', linewidth=1.5, label=r'$t_{es}$')
        ax.axvline(x=t_ls, color='black', linestyle='-', linewidth=1.5, label=r'$t_{ls}$')
        ax.axvline(x=t_le, color='black', linestyle='-', linewidth=1.5, label=r'$t_{le}$')
        
        max_y = max([e[1] for e in self.energy_profile]) if self.energy_profile else 0
        ax.text(t_es, max_y * 1.1, r'$t_{es}$', fontsize=10, ha='center')
        ax.text(t_ls, max_y * 1.1, r'$t_{ls}$', fontsize=10, ha='center')
        ax.text(t_le, max_y * 1.1, r'$t_{le}$', fontsize=10, ha='center')
        
        # Add legend for clarity
        ax.legend()
 
        plt.tight_layout()
        plt.show()