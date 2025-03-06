from datetime import datetime, timedelta
from config import config
from helpers import convert_hour_to_datetime
import flexoffer_logic 
from typing import List
from classes.flexOffer import FlexOffer

def startAlignment(fos: List[FlexOffer]) -> FlexOffer:
    assert fos
    earliest_start_times: List[datetime] = []
    durations: List[int] = []
    for fo in fos:
        earliest_start_times.append(fo.get_earliest)
        durations.append(len(fo.energy_profile))
    est = min(earliest_start_times)
    lst = est + timedelta(hours=min(durations))
    
    afo: FlexOffer = fos[0]
    for element in fos[1:-1]:
        delta:timedelta = afo.get_earliest - element.get_earliest
        diffInTime: float = delta.total_seconds()
        if diffInTime != 0: diffInTime /= config.TIME_RESOLUTION
        afo.plus(diffInTime, element.energy_profile)
    
    afo.earliest_start = est
    afo.end_time = lst + timedelta(len(afo.energy_profile))

    return afo




def start_alignment_fast(flex_offers: List[FlexOffer]) -> FlexOffer:
    # First extract the data
    min_profiles = [fo.get_min_profile for fo in flex_offers]
    max_profiles = [fo.get_max_profile for fo in flex_offers]
    earliest = [fo.get_earliest.hour for fo in flex_offers]
    latest = [fo.get_latest.hour for fo in flex_offers]

    offsets = earliest
    
    # Use the data in the cpp function
    result = flexoffer_logic.start_alignment_aggregate(min_profiles, max_profiles, earliest, latest, offsets)
    



    aggregated_min = list(result["aggregated_min"])
    aggregated_max = list(result["aggregated_max"])
    global_earliest = result["global_earliest"]
    aggregated_latest = result["aggregated_latest"]
    common_length = result["common_length"]
    
    new_earliest = convert_hour_to_datetime(global_earliest)
    new_latest = convert_hour_to_datetime(aggregated_latest)
    new_duration = timedelta(hours=common_length)
    new_end = new_latest + new_duration
    
    aggregated_profile = list(zip(aggregated_min, aggregated_max))
    
    # Create a new FlexOffer.
    aggregated_offer = FlexOffer(
        offer_id=-1,
        earliest_start=new_earliest,
        latest_start=new_latest,
        end_time=new_end,
        duration=new_duration,
        energy_profile=aggregated_profile,
        min_energy=sum(aggregated_min),
        max_energy=sum(aggregated_max)
    )
    
    return aggregated_offer