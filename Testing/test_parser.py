from database.parser import fo_parser
import datetime
from config import config

CONFIG = config()


def test_foParserAllData():
    result = fo_parser()
    for element in result:
        element.print_flexoffer()


def test_foParserRangeData():
    dt1 = (datetime.datetime(2030, 6, 30, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    dt2 = (datetime.datetime(2031, 6, 30, 0, 0) - datetime.datetime(2020, 1, 1, 0, 0)).total_seconds() / CONFIG.TIME_RESOLUTION
    result = fo_parser(dt1, dt2)
    for element in result:
        element.print_flexoffer()
