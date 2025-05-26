from datetime import datetime, timedelta
import pandas as pd
import time

def convert_hour_to_datetime(hour: int) -> datetime:
    return datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)

def dt_to_unix(dt):
    return int(dt.timestamp())

def dt_to_unix_seconds(dt_obj):
    return int(time.mktime(dt_obj.timetuple())) 


def sol_to_df(sol):
    """
    Convert a solution dict into a pandas DataFrame.
    Handles missing variables (e.g., if pr_up or pb_up aren't present).
    """
    all_vars = list(sol.keys())
    records = []

    for var in all_vars:
        agent_dict = sol[var]
        for agent, times in agent_dict.items():
            for t, v in times.items():
                records.append({
                    'time_slot': t,
                    'agent': agent,
                    'var': var,
                    'value': v
                })

    df = pd.DataFrame(records)

    if df.empty:
        return pd.DataFrame(columns=["time_slot", "agent"] + all_vars)

    # Pivot: one row per (time, agent), one column per var
    df = df.pivot_table(
        index=["time_slot", "agent"],
        columns="var",
        values="value",
        fill_value=0,
        aggfunc="first"
    ).reset_index()

    return df

def add_spot_prices_to_df(df, spot_series):
    """
    Add the corresponding spot price for each time slot.
    Assumes time_slot is an integer index and spot_series is a pandas Series with integer index.
    """
    # If spot_series has a datetime index, convert to integer slot indices first
    if isinstance(spot_series.index[0], pd.Timestamp):
        spot_series = spot_series.reset_index(drop=False)
        spot_series["time_slot"] = range(len(spot_series))
        spot_df = spot_series[["time_slot", "SpotPriceDKK"]].rename(columns={"SpotPriceDKK": "spot_price"})
    else:
        spot_df = pd.DataFrame({
            "time_slot": spot_series.index,
            "spot_price": spot_series.values
        })

    df_with_price = df.merge(spot_df, on="time_slot", how="left")
    return df_with_price


def round_datetime_to_resolution(dt: datetime, resolution_seconds: int, direction: str = "down") -> datetime:
    seconds_since_hour = (dt - dt.replace(minute=0, second=0, microsecond=0)).total_seconds()
    if direction == "down":
        return dt - timedelta(seconds=seconds_since_hour % resolution_seconds)
    elif direction == "up":
        delta = resolution_seconds - (seconds_since_hour % resolution_seconds) if seconds_since_hour % resolution_seconds > 0 else 0
        return dt + timedelta(seconds=delta)
    else:
        raise ValueError("direction must be 'down' or 'up'")