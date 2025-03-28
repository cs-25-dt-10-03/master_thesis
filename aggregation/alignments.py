from datetime import datetime, timedelta
from config import config
from helpers import convert_hour_to_datetime
import flexoffer_logic
from flexoffer_logic import Flexoffer, TimeSlice, start_alignment_aggregate, balance_alignment_aggregate
from typing import List
#from classes.flexOffer import FlexOffer

def startAlignment(fos: List[Flexoffer]) -> Flexoffer:
    assert fos
    earliest_start_times: List[datetime] = []
    durations: List[int] = []
    for fo in fos:
        earliest_start_times.append(fo.get_earliest)
        durations.append(len(fo.energy_profile))
    est = min(earliest_start_times)
    lst = est + timedelta(hours=min(durations))
    
    afo: Flexoffer = fos[0]
    for element in fos[1:-1]:
        delta:timedelta = afo.get_earliest - element.get_earliest
        diffInTime: float = delta.total_seconds()
        if diffInTime != 0: diffInTime /= config.TIME_RESOLUTION
        afo.plus(diffInTime, element.energy_profile)
    
    afo.earliest_start = est
    afo.end_time = lst + timedelta(len(afo.energy_profile))

    return afo


def start_alignment_fast(flex_offers: List[Flexoffer]) -> Flexoffer:
    afo = start_alignment_aggregate(flex_offers)
    return afo

def balance_alignment_fast(flex_offers: List[Flexoffer]) -> Flexoffer:
    afo = balance_alignment_aggregate(flex_offers)
    return afo