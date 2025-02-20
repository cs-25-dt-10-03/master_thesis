from datetime import datetime, timedelta
from config import config
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class flexOffer:
    def __init__(self, offer_id: str,
                 earliest_start: datetime,
                 end_time: datetime,
                 energy_profile: List[Tuple[float, float]],
                 min_energy: Optional[float] = None,
                 max_energy: Optional[float] = None,
                 total_energy_limit: Optional[float] = None):
        
        self.offer_id = offer_id
        self.earliest_start = earliest_start
        self.end_time=end_time
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
    def get_end(self):
        return self.end_time.replace(minute=0, second=0, microsecond=0)
    
    def __repr__(self):
        return (f"<FlexOffer id={self.offer_id} "
                f"start_window=({self.get_earliest} - {self.get_end}) "
                f"duration={self.get_end - self.get_earliest } total_energy={self.total_energy}>")

    def plot(self, schedule_start: Optional[datetime] = None, show_window: bool = True):
        if schedule_start is None:
            schedule_start = self.earliest_start

        print(schedule_start)
        print(self.end_time)

        num_slots = len(self.energy_profile)
        if num_slots == 0:
            raise ValueError("Energy profile is empty, nothing to plot.")

        # Compute time step
        time_slot_resolution = timedelta(minutes = config.TIME_RESOLUTION)

        times = [schedule_start + i * time_slot_resolution for i in range(num_slots)]
        times_num = [mdates.date2num(t) for t in times]

        min_values = [e[0] for e in self.energy_profile]
        flexible_values = [e[1] - e[0] for e in self.energy_profile]

        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot minimum energy in blue, flexible energy in orange
        ax.bar(times_num, min_values, width=1/24, align='edge', color='blue', edgecolor='black', label='Minimum Energy')
        ax.bar(times_num, flexible_values, width=1/24, bottom=min_values, align='edge', color='orange', edgecolor='black', label='Flexible Energy')

        # Shade the charging window (earliest start to charging end)
        if show_window:
            window_left = mdates.date2num(schedule_start)
            window_right = mdates.date2num(self.end_time)
            ax.axvspan(window_left, window_right, color='gray', alpha=0.2, label="Charging Window")

        # Set labels and formatting
        ax.set_title(f"FlexOffer {self.offer_id}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Energy (kWh)")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()

        # Plot key time markers
        t_es = mdates.date2num(self.earliest_start)
        t_le = mdates.date2num(self.end_time)

        ax.axvline(x=t_es, color='black', linestyle='--', linewidth=1.5, label=r'$t_{es}$ (Earliest Start)')
        ax.axvline(x=t_le, color='green', linestyle='--', linewidth=1.5, label=r'$t_{le}$ (End of Window)')

        # Add text annotations
        max_y = max(e[1] for e in self.energy_profile) if self.energy_profile else 0
        ax.text(t_es, max_y * 1.1, r'$t_{es}$', fontsize=10, ha='center')
        ax.text(t_le, max_y * 1.1, r'$t_{le}$', fontsize=10, ha='center', color='green')

        # Add legend for clarity
        ax.legend()

        plt.tight_layout()
        plt.show()
