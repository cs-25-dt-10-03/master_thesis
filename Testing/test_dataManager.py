from database.dataManager import fetchAllMfrrPrices, fetchAllSpotPrices
from json import dumps


def test_manager():
    result = fetchAllSpotPrices()
    print(dumps(result, indent=4))