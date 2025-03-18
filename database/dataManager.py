import pandas as pd
from typing import List
import os
from datetime import datetime
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

def load_data():
    df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Electricity prices.csv"), dtype="unicode", delimiter=",", skiprows=0)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.round("h")
    return df

def get_price_at_datetime(datetime_value):
    df = load_data()
    datetime_value = pd.to_datetime(datetime_value, unit="s")
    row = df.loc[df['Timestamp'] == datetime_value]
    if not row.empty:
        return float(row['Spot Price [DKK/kWh]'].values[0])
    return None

def get_prices_in_range(start_timestamp, end_timestamp):
    df = load_data()
    start_datetime = pd.to_datetime(start_timestamp, unit="s")
    end_datetime = pd.to_datetime(end_timestamp, unit="s")
    mask = (df['Timestamp'] >= start_datetime) & (df['Timestamp'] <= end_datetime)
    
    return df.loc[mask, 'Spot Price [DKK/kWh]'].astype(float).tolist()



def fetch_mFRR_by_date(target_date):
    df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR.csv"), delimiter=";")
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")

    target_date = pd.to_datetime(target_date, unit="s")
    corresponding_row = df[df["HourDK"] == target_date]

    print(f"target date {target_date}")
    print(f"df selected date {corresponding_row}")

    return corresponding_row


def fetch_mFRR_by_range(start_date, end_date):
    df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR.csv"), delimiter=";")
    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    start_date = pd.to_datetime(start_date, unit="s")
    end_date = pd.to_datetime(end_date, unit="s")

    mask = (df["HourDK"] >= start_date) & (df["HourDK"] <= end_date)
    df_filtered = df[mask]
    return df_filtered