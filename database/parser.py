from dataManager import getEVsInRange, fetchEvData, fetchEvModels

from typing import List
import pandas as pd
import flexoffer_logic as fo


def fo_parser(start_hour: int = 0, end_hour: int = 0) -> List[fo.Flexoffer]:
    if start_hour == 0 and end_hour == 0:
        return all_data()
    else:
        return range_data()


def all_data() -> List[fo.Flexoffer]:
    evModels = fetchEvModels()
    data: List[pd.DataFrame] = fetchEvData()

    result: List[fo.Flexoffer] = []

    for i, df in enum(data):

    return result


def range_data(start_hour: int, end_hour: int) -> List[fo.Flexoffer]:
    data: List[pd.DataFrame] = getEVsInRange(start_hour, end_hour)
    result: List[fo.Flexoffer] = []

    return result
