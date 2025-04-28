import datetime

from database.dataManager import fetch_mFRR_by_range, getEvAtDatetime, getEVsInRange, fetchEvModels, convertYearInfo
from config import config

CONFIG = config()


def test_fetch_mfrr_by_range():
    timestamp = 1710435600
    timestamp2 = 1710439200
    result = fetch_mFRR_by_range(timestamp, timestamp2)
    print(result)


def test_getEvAtDatetime():
    timestamp = (datetime.datetime(2023, 11, 19, 1, 0, 7) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    print(timestamp)
    result = getEvAtDatetime(round(timestamp))
    print(result)


def test_getEVsInRange():
    dt1 = (datetime.datetime(2023, 11, 19, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    dt2 = (datetime.datetime(2023, 11, 21, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    result = getEVsInRange(dt1, dt2)
    print(result)


def test_fetchEvModels():
    fetchEvModels()


def test_convertYearInfo():
    dt1 = (datetime.datetime(2028, 11, 19, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    dt2 = (datetime.datetime(2029, 11, 21, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    result = getEVsInRange(dt1, dt2)
    print("Len is: ", len(result))
    print("First is: ", result[0])
    convertYearInfo(result)
    print(result[0])
