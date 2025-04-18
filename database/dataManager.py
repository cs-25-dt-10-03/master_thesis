import pandas as pd
from typing import List
import os
from config import config


def fetchEvData() -> List[pd.DataFrame]:
    data = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Household data.csv"), delimiter=",", skiprows=1, dtype=str, low_memory=False)
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
            ev_df['Passed Hours'] = ev_df['Passed Hours'].astype(float)
            ev_df['EV model'] = ev_df['EV model'].loc[0]
            dfs.append(ev_df)
        i += 6

    return dfs


def getEVsInRange(start_hour: int, end_hour: int) -> List[pd.DataFrame]:
    dfs = fetchEvData()
    result: List[pd.DataFrame] = []
    for ev in dfs:
        mask = (ev['Passed Hours'] >= start_hour) & (ev['Passed Hours'] <= end_hour)
        result.append(ev.loc[mask])

    return result


def getEvAtDatetime(datetime_value: int) -> List[pd.DataFrame]:
    dfs = fetchEvData()
    result: List[pd.DataFrame] = []

    for ev in dfs:
        result.append(ev[round(ev['Passed Hours']) == datetime_value])

    if result:
        return result

    return None



def loadSpotPriceData():
    if config.TIME_RESOLUTION == 3600:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "ElspotPrices.csv"), dtype="unicode", delimiter=",", skiprows=0)
    else:
        print("Hiya papaya!")
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "ElspotPrices_15min.csv"), dtype="unicode", delimiter=",", skiprows=0)
    df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
    df['SpotPriceDKK'] = pd.to_numeric(df['SpotPriceDKK'], errors='coerce')
    return df


def get_price_at_datetime(datetime_value):
    df = loadSpotPriceData()
    datetime_value = pd.to_datetime(datetime_value, unit="s")
    row = df.loc[df['HourDK'] == datetime_value]
    if not row.empty:
        return float(row['SpotPriceDKK'].values[0])
    return None


def get_prices_in_range(start_timestamp, end_timestamp):

    df = loadSpotPriceData()
    start_datetime = pd.to_datetime(start_timestamp, unit="s")
    end_datetime = pd.to_datetime(end_timestamp, unit="s")
    print(f"Starttt:DARE TIME ::: { start_datetime}, \n {end_datetime}")
    mask = (df['HourDK'] >= start_datetime) & (df['HourDK'] <= end_datetime)
    filtered_df = df.loc[mask]
    filtered_df = filtered_df.set_index('HourDK')
    filtered_df.drop(columns=['HourUTC'], inplace=True)
    return filtered_df


def fetch_mFRR_by_date(target_date):
    if config.TIME_RESOLUTION == 3600:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR.csv"), delimiter=",")
    else:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR_15min.csv"), delimiter=",")
    df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
    target_date = pd.to_datetime(target_date, unit="s")
    corresponding_row = df[df["HourDK"] == target_date]

    return corresponding_row


def fetch_mFRR_by_range(start_date, end_date):
    if config.TIME_RESOLUTION == 3600:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR.csv"), delimiter=",")
    else:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR_15min.csv"), delimiter=",")

    start_date = pd.to_datetime(start_date, unit="s")
    end_date = pd.to_datetime(end_date, unit="s")
    df['HourDK'] = pd.to_datetime(df['HourDK'], errors='coerce')
    df['mFRR_UpPriceDKK'] = pd.to_numeric(df['mFRR_UpPriceDKK'], errors='coerce')
    df['mFRR_DownPriceDKK'] = pd.to_numeric(df['mFRR_DownPriceDKK'], errors='coerce')

    mask = (df["HourDK"] >= start_date) & (df["HourDK"] <= end_date)
    df_filtered = df[mask]
    df_filtered = df_filtered.set_index('HourDK')
    df_filtered.drop(columns=['HourUTC'], inplace=True)
    return df_filtered


def fetch_Regulating_by_range(start_date, end_date):
    if config.TIME_RESOLUTION == 3600:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Regulating.csv"), delimiter=",")
    else:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Regulating_15min.csv"), delimiter=",")

    df["HourDK"] = pd.to_datetime(df["HourDK"], errors="coerce")
    start_date = pd.to_datetime(start_date, unit="s")
    end_date = pd.to_datetime(end_date, unit="s")

    mask = (df["HourDK"] >= start_date) & (df["HourDK"] <= end_date)
    df_filtered = df[mask]
    df_filtered = df_filtered.set_index('HourDK')
    df_filtered.drop(columns=['HourUTC'], inplace=True)
    return df_filtered

