import pandas as pd
from database.dataManager import loadSpotPriceData
from database.mfRRPriceData import load_mfrr_price_data  # assuming this function exists
from datetime import datetime, timedelta

def get_hourly_spot_prices(start, duration):
    df = loadSpotPriceData()
    start_dt = pd.to_datetime(start, unit="s")
    end_dt = start_dt + timedelta(hours=duration)
    mask = (df["Timestamp"] >= start_dt) & (df["Timestamp"] < end_dt)
    return df.loc[mask, "Spot Price [DKK/kWh]"].astype(float).tolist()

def get_hourly_mfrr_prices(start, duration):
    df = load_mfrr_price_data()
    start_dt = pd.to_datetime(start, unit="s")
    end_dt = start_dt + timedelta(hours=duration)
    mask = (df["HourDK"] >= start_dt) & (df["HourDK"] < end_dt)
    return {
        "up": df.loc[mask, "mFRR_up"].astype(float).tolist(),
        "down": df.loc[mask, "mFRR_down"].astype(float).tolist()
    }