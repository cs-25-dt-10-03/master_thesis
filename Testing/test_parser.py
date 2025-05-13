from database.parser import fo_parser
import datetime
from config import config

CONFIG = config()


def test_foParserAllData():
    result = fo_parser()
    for element in result:
        element.print_flexoffer()


def test_foParserRangeData():
    start = (datetime.datetime(2024, 3, 1, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    stop = (datetime.datetime(2024, 3, 30, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    result = fo_parser(start, stop)
    for element in result:
        element.print_flexoffer()
