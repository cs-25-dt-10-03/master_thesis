from database.dataManager import fetchEvData, get_price_at_datetime, get_prices_in_range
from json import dumps
from config import config
from datetime import datetime, timedelta

def test_manager():
    timestamp = 1710435600
    timestamp2 = 1710439200
    result = get_price_at_datetime(timestamp)
    print(result)