from datetime import datetime, timedelta
import pandas as pd
import time
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
    
def filter_day_offers(flexoffers: List[Any], dfos: List[Any],sim_start_ts: int, day: int, slots_per_day: int) -> Tuple[List[Any], List[Any], int, int]:
    """
    Returns (active_fos, active_dfos, start_slot, end_slot) for the given calendar day.
    """
    start_slot = day * slots_per_day + 17 * 3600 // config.TIME_RESOLUTION
    end_slot   = start_slot + slots_per_day

    active_fos = [
        fo for fo in flexoffers
        if ((fo.get_est() - sim_start_ts) // config.TIME_RESOLUTION) < end_slot
        and (((fo.get_est() - sim_start_ts) // config.TIME_RESOLUTION) + fo.get_duration()) > start_slot
    ]
    active_dfos = [
        dfo for dfo in dfos
        if ((dfo.get_est() - sim_start_ts) // config.TIME_RESOLUTION) < end_slot
        and (((dfo.get_est() - sim_start_ts) // config.TIME_RESOLUTION) + dfo.get_duration()) > start_slot
    ]

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
