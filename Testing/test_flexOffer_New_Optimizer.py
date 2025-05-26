import pytest
from config import config
import pandas as pd
from datetime import datetime, timedelta
from optimization.test_optimizers import FO_Opt
from flexoffer_logic import Flexoffer, TimeSlice, set_time_resolution

@pytest.fixture
def sample_flexoffer():
    now = datetime(2024, 1, 16, 0, 0, 0)
    est = int(now.timestamp())
    lst = int((now + timedelta(hours=1)).timestamp())
    et  = int((now + timedelta(hours=3)).timestamp())
    duration = 2

    profile = [TimeSlice(0.0, 11.0), TimeSlice(0.0, 11.0)]
    fo = Flexoffer(
        offer_id=1,
        earliest_start=est,
        latest_start=lst,
        end_time=et,
        profile=profile,
        duration=duration,
        min_overall_alloc=13.0,
        max_overall_alloc=20.0
    )
    return fo

@pytest.fixture
def spot_prices():
    sim_start_ts = pd.to_datetime(config.SIMULATION_START_DATE)

    index = pd.date_range(start=sim_start_ts, periods=4, freq='h')
    prices = pd.Series([300, 200, 100, 50], index=index)
    return prices


def test_if_schedule_is_set(sample_flexoffer, spot_prices):
    
    optimizer = FO_Opt([sample_flexoffer], spot_prices)
    optimizer.run()
    assert sample_flexoffer.get_scheduled_allocation() is not None
     
    # sample_flexoffer.print_flexoffer()
    # print(spot_prices)