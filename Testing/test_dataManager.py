from database.dataManager import fetchEvData, get_price_at_datetime, get_prices_in_range, fetch_mFRR_by_date, load_mFRR_data
from json import dumps
from config import config
from datetime import datetime, timedelta

def test_manager():
    timestamp = 1710435600
    result = fetch_mFRR_by_date(timestamp)
    print(result)