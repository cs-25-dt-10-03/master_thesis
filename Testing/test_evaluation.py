import os
from config import config
from evaluation.evaluation_pipeline import evaluate_configurations, get_scenarios
from database.dataManager import load_and_prepare_prices
from datetime import timedelta, datetime

def test_evaluation():
    start = config.SIMULATION_START_DATE
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )

    # 2. Evaluate all scenarios
    print("Starting evaluation...")
    evaluate_configurations(spot, reserve, activation, indicators)

    print("Evaluation finished!")

