from database.dataManager import getEVsInRange, fetchEvData, fetchEvModels
from classes.electricVehicle import ElectricVehicle

from typing import List
import pandas as pd
import flexoffer_logic as fo


def fo_parser(start_hour: int = 0, end_hour: int = 0) -> List[fo.Flexoffer]:
    if start_hour == 0 and end_hour == 0:
        return all_data()
    else:
        return range_data()


def all_data() -> List[fo.Flexoffer]:
    evModels: pd.DataFrame = fetchEvModels()
    data: List[pd.DataFrame] = fetchEvData()

    result: List[fo.Flexoffer] = []

    id: int = 0
    for df in data:
        ev_model = df['EV model'].loc[0]
        connected = False
        start = None
        stop = None
        soc = 0.0
        for _, row in df.iterrows():
            if not connected and row['EV state'] != 'Driving':
                connected = True
                start = row['Passed Hours']
                soc = float(row['SoC'])
            elif connected and row['EV state'] == 'Driving':
                connected = False
                stop = row['Passed Hours'] - 1
            if start is not None and stop is not None:
                if soc == 1.0:
                    continue
                model = evModels.loc[evModels['EV model'] == ev_model]
                ev = ElectricVehicle(id, model['Battery capacity [kWh]'].astype(float).values[0], 0.0, 1.0, model['Charging power [kW]'].astype(float).values[0], 1.0, soc)
                if start + 1 < stop:
                    result.append(ev.create_flexoffer(start, stop))
                    id += 1
                start = None
                stop = None


    return result


def range_data(start_hour: int, end_hour: int) -> List[fo.Flexoffer]:
    data: List[pd.DataFrame] = getEVsInRange(start_hour, end_hour)
    result: List[fo.Flexoffer] = []

    return result
