from datetime import datetime, timedelta
import pandas as pd
import time
from flexoffer_logic import Flexoffer, DFO
from config import config
from typing import Dict, List, Tuple, Any

def convert_hour_to_datetime(hour: int) -> datetime:
    return datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)

def dt_to_unix(dt):
    return int(dt.timestamp())

def dt_to_unix_seconds(dt_obj):
    return int(time.mktime(dt_obj.timetuple())) 

def round_datetime_to_resolution(dt: datetime, resolution_seconds: int, direction: str = "down") -> datetime:
    seconds_since_hour = (dt - dt.replace(minute=0, second=0, microsecond=0)).total_seconds()
    if direction == "down":
        return dt - timedelta(seconds=seconds_since_hour % resolution_seconds)
    elif direction == "up":
        delta = resolution_seconds - (seconds_since_hour % resolution_seconds) if seconds_since_hour % resolution_seconds > 0 else 0
        return dt + timedelta(seconds=delta)
    else:
        raise ValueError("direction must be 'down' or 'up'")
    
def filter_day_offers(flexoffers, dfos, sim_start_ts: int, day: int, slots_per_day: int):
    """
    Returns (active_fos, active_dfos, start_slot, end_slot) for the given calendar day,
    correctly including any FO that overlaps the day—even if it crosses midnight.
    """
    # 1) compute the slot indices of this day’s start/end (relative to sim_start_ts)
    start_slot = day * slots_per_day
    end_slot   = start_slot + slots_per_day

    # 2) turn those into absolute timestamps
    day_start_ts = sim_start_ts + start_slot * config.TIME_RESOLUTION
    day_end_ts   = sim_start_ts + end_slot   * config.TIME_RESOLUTION

    def overlaps(fo):
        # FO window is [fo.get_est(), fo.get_et()]
        return (fo.get_est() <  day_end_ts) \
           and (fo.get_et()  >  day_start_ts)

    active_fos  = [fo for fo in flexoffers if overlaps(fo)]
    active_dfos = [dfo for dfo in dfos      if overlaps(dfo)]

    return active_fos, active_dfos, start_slot, end_slot


def slice_prices(prices: Dict[str, Any], start_slot: int, end_slot: int):
    """
    Unpack and slice all market series for [start_slot:end_slot].
    """
    spot       = prices["spot"]      [start_slot:end_slot].reset_index(drop=True)
    reserve    = prices["reserve"]   [start_slot:end_slot].reset_index(drop=True)
    activation = prices["activation"][start_slot:end_slot].reset_index(drop=True)
    indic      = prices["indicators"][start_slot:end_slot]
    imbalance  = prices["imbalance"] [start_slot:end_slot].reset_index(drop=True)


    expected = end_slot - start_slot
    assert len(spot) == expected, (
        f"[slice_prices] got {len(spot)} slots, expected {expected}"
    )
    print(f"[slice_prices] slots {start_slot}–{end_slot} ",
          f"→ {spot.index[0]} … {spot.index[-1]}")

    return spot, reserve, activation, indic, imbalance


def format_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    


def print_flexoffer_energy(offers: list[Any]) -> None:

    if isinstance(offers[0], Flexoffer):
        for i, fo in enumerate(offers):

            print(f"--- AFO {i} ---")
            print(f"Min Overall Alloc: {fo.get_min_overall_alloc():.2f} kWh")
            # print(f"Earliest Start    (est): {format_ts(fo.get_est())}")
            # print(f"Latest Start      (lst): {format_ts(fo.get_lst())}")
            # print(f"End Time          (et):  {format_ts(fo.get_et())}")
    
    elif isinstance(offers[0], DFO):
        for i, fo in enumerate(offers):

            print(f"--- AFO {i} ---")
            print(f"Min Overall Alloc: {fo.get_min_total_energy():.2f} kWh")