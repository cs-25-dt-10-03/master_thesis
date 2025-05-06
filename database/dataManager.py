import pandas as pd
import datetime
from typing import List
import os
import datetime
from config import config
import numpy as np


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
            ev_df = ev_df.loc[ev_df['EV model'] != 'no EV']
            ev_df = ev_df.dropna(subset=['EV state', 'SoC'])
            ev_df.reset_index(drop=True, inplace=True)
            if not ev_df.empty:
                dfs.append(ev_df)
        i += 6

    return dfs


def fetchEvModels() -> pd.DataFrame:
    return pd.read_csv(os.path.join(config.DATA_FILEPATH, "EV Models.csv"), delimiter=",", dtype=str, low_memory=False)


def getEVsInRange(start_hour: int, end_hour: int) -> List[pd.DataFrame]:
    dfs = fetchEvData()
    result: List[pd.DataFrame] = []
    for ev in dfs:
        mask = (ev['Passed Hours'] >= start_hour) & (ev['Passed Hours'] <= end_hour)
        pruned_ev = ev.loc[mask]
        if not pruned_ev.empty:
            pruned_ev.reset_index(drop=True, inplace=True)
            result.append(pruned_ev)

    return result


def getEvAtDatetime(datetime_value: int) -> List[pd.DataFrame] | None:
    dfs: List[pd.DataFrame] = fetchEvData()
    result = []

    for ev in dfs:
        result.append(ev[round(ev['Passed Hours']) == datetime_value])

    if result:
        return result

    return None


def loadSpotPriceData():
    if config.TIME_RESOLUTION == 3600:
        df = pd.read_csv(os.path.join(config.DATA_FILEPATH, "ElspotPrices.csv"), dtype="unicode", delimiter=",", skiprows=0)
    else:
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
    print(f"Starttt:DARE TIME ::: {start_datetime}, \n {end_datetime}")
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


# --- Har lige lavet en data loader der læser alle priser sammen. Gør det lidt nemmere for optimizeren ---
def load_and_prepare_prices(start_ts, horizon_slots, resolution):
    """
    Load spot and mFRR CSV, then slice exact horizon.
    start_ts: pandas.Timestamp matching CSV index
    horizon_slots: number of slots to simulate
    resolution: pandas frequency string, e.g. 'H' for hourly
    """

    if resolution == 3600:
        spot = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Elspotprices.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'SpotPriceDKK'])
        mfrr = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'mFRR_UpPriceDKK', 'mFRR_DownPriceDKK'])
        act = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Regulating.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'BalancingPowerPriceUpDKK', 'BalancingPowerPriceDownDKK'])
    else:
        spot = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Elspotprices_15min.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'SpotPriceDKK'])
        mfrr = pd.read_csv(os.path.join(config.DATA_FILEPATH, "mFRR_15min.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'mFRR_UpPriceDKK', 'mFRR_DownPriceDKK'])
        act = pd.read_csv(os.path.join(config.DATA_FILEPATH, "Regulating_15min.csv"), parse_dates=['HourDK'], usecols=['HourDK', 'BalancingPowerPriceUpDKK', 'BalancingPowerPriceDownDKK'])

    spot = spot.drop_duplicates(subset='HourDK')
    mfrr = mfrr.drop_duplicates(subset='HourDK')
    act = act.drop_duplicates(subset='HourDK')

    # if not isinstance(start_ts, datetime.datetime):
    start_ts = pd.to_datetime(start_ts)
    end_ts = start_ts + pd.to_timedelta(horizon_slots * resolution, unit="s")


    spot = spot[(spot["HourDK"] >= start_ts) & (spot["HourDK"] < end_ts)]
    mfrr = mfrr[(mfrr["HourDK"] >= start_ts) & (mfrr["HourDK"] < end_ts)]
    act = act[(act["HourDK"] >= start_ts) & (act["HourDK"] < end_ts)]

    spot.set_index('HourDK', inplace=True)
    mfrr.set_index('HourDK', inplace=True)
    act.set_index('HourDK', inplace=True)

    spot_prices = spot['SpotPriceDKK']
    reserve_prices = mfrr[['mFRR_UpPriceDKK', 'mFRR_DownPriceDKK']]
    activation_prices = act[['BalancingPowerPriceUpDKK', 'BalancingPowerPriceDownDKK']]

    delta_up = (activation_prices['BalancingPowerPriceUpDKK'] > np.nanmean(activation_prices['BalancingPowerPriceUpDKK'])).astype(int)
    delta_dn = (activation_prices['BalancingPowerPriceDownDKK'] > np.nanmean(activation_prices['BalancingPowerPriceDownDKK'])).astype(int)
    indicators = list(zip(delta_up, delta_dn))

    return spot_prices, reserve_prices, activation_prices, indicators


def convertYearInfo(df: List[pd.DataFrame]) -> None:
    """
    df bliver passed som reference, så den bliver ændret inplace
    """
    for element in df:
        element['Timestamp'] = pd.to_datetime(element['Timestamp'], format="%b %d, %Y, %I:%M:%S %p")
        element['Timestamp'] = [time.replace(year=2024) for time in element['Timestamp']]
        element['Passed Hours'] = [round((time - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / config.TIME_RESOLUTION) for time in element['Timestamp']]
    return
