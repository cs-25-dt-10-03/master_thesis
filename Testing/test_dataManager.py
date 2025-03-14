from database.dataManager import fetchAllMfrrPrices, fetchAllSpotPrices, fetchSpotPricesByDate, fetchSpotPricesInRange
from json import dumps
from config import config

def test_manager():
    timestamp = 1710435600
    timestamp2 = 1710539200
    result = fetchSpotPricesInRange(timestamp, timestamp2)
    print(dumps(result, indent=4))