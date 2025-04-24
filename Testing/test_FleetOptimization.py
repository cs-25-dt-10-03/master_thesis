from optimization.simulator import create_aggregated_offers
from evaluation.eval import run_evaluation
from flexoffer_logic import Flexoffer, TimeSlice
from optimization.flexOfferOptimizer import schedule_offers
from database.dataManager import load_and_prepare_prices
import pandas as pd
from config import config
import os

DATA_PATH = os.path.dirname(__file__)

start_ts = pd.Timestamp("2024-03-01 00:00:00")
horizon = 24

spot, reserve, activation, indicators = load_and_prepare_prices(
    os.path.join(config.DATA_FILEPATH, "ElspotPrices.csv"),
    os.path.join(config.DATA_FILEPATH, "mFRR.csv"),
    os.path.join(config.DATA_FILEPATH, "Regulating.csv"),
    start_ts,
    horizon,
    resolution="h"
)

# to example with 2 offers
min_lists = [ [0]*horizon, [5]*horizon ]
max_lists = [ [10]*horizon, [15]*horizon ]
offers = create_aggregated_offers(min_lists, max_lists)

solution = schedule_offers(
    offers, spot, reserve, activation, indicators
)

df = run_evaluation(min_lists, max_lists, spot, reserve, activation, indicators)
print(df)
