import datetime

from database.dataManager import fetch_mFRR_by_date, getEvAtDatetime, getEVsInRange
from config import config

CONFIG = config()


def test_fetch_mfrr_by_date():
    timestamp = 1710435600
    result = fetch_mFRR_by_date(timestamp)
    print(result)


def test_getEvAtDatetime():
    timestamp = (datetime.datetime(2023, 11, 19, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)) / CONFIG.TIME_RESOLUTION
    result = getEvAtDatetime(timestamp)
    print(result)


def test_getEVsInRange():
    dt1 = (datetime.datetime(2023, 11, 19, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    dt2 = (datetime.datetime(2024, 11, 19, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    result = getEVsInRange(dt1, dt2)
    print(result)
