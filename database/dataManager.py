import pandas as pd
from typing import List
import os
from config import config


def fetchEvData() -> pd.core.frame.DataFrame:
    data = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Household data.csv"), dtype="unicode", delimiter=",", skiprows=1)
    column_names = data.columns.tolist()

    dfs: List[pd.DataFrame] = []
    i = 7
    while i < len(column_names):
        if "EV model" in column_names[i]:
            ev_df = data.iloc[:, [0, 1, i, i+1, i+2, i+3, i+4, i+5]].copy()
            ev_df.columns = [
                    "Passed Hours",
                    "Timestamp",
                    "EV model",
                    "Charging strategy",
                    "EV state",
                    "SoC",
                    "Driving distance",
                    "Times when soc not satisfied"
                    ]
            dfs.append(ev_df)
        i += 1

    return dfs
