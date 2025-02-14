from datetime import datetime, timedelta
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
 
        fig, ax = plt.subplots(figsize=(7, 3))


        ax.bar(times_num, self.energy_profile, width=1/24, align='edge', color='skyblue', edgecolor='black')


        if show_window:
            window_left = mdates.date2num(self.earliest_start)
            window_right = mdates.date2num(self.latest_start)
            ax.axvspan(window_left, window_right, color='gray', alpha=0.2)

        ax.set_title(f"FlexOffer {self.offer_id}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Energy (kwh)")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        fig.autofmt_xdate()

        #These lines are for setting times black bars
        t_es = mdates.date2num(self.earliest_start)
        t_ls = mdates.date2num(self.latest_start)
        t_le = mdates.date2num(end_time)
        ax.axvline(x=t_es, color='black', linestyle='-', linewidth=1.5, label=r'$t_{es}$')
        ax.axvline(x=t_ls, color='black', linestyle='-', linewidth=1.5, label=r'$t_{ls}$')
        ax.axvline(x=t_le, color='black', linestyle='-', linewidth=1.5, label=r'$t_{le}$')
        ax.text(t_es, max(self.energy_profile) * 1.1, r'$t_{es}$', fontsize=10, ha='center')
        ax.text(t_ls, max(self.energy_profile) * 1.1, r'$t_{ls}$', fontsize=10, ha='center')
        ax.text(t_le, max(self.energy_profile) * 1.1, r'$t_{le}$', fontsize=10, ha='center')

        plt.tight_layout()
        plt.show()