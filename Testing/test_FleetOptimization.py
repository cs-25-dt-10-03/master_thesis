import os
from config import config
from evaluation.run_evaluation import evaluate_configurations ,get_scenarios
from database.dataManager import load_and_prepare_prices

def test_evaluation():
    start_ts = config.SIMULATION_START_DATE
    slots_per_day = int(24 * (3600 / config.TIME_RESOLUTION))
    horizon_slots = config.SIMULATION_DAYS * slots_per_day

    print(f"START TID: {start_ts} \n")
    print(f"slots per dag: {slots_per_day} \n")
    print(f"Hvor man slots i alt: {horizon_slots} \n")


    spot, reserve, activation, indicators = load_and_prepare_prices(
        start_ts=start_ts,
        horizon_slots=horizon_slots,
        resolution=config.TIME_RESOLUTION
    )

    # 2. Evaluate all scenarios
    print("Starting evaluation...")
    evaluate_configurations(spot, reserve, activation, indicators)

    # (Optional) 3. Plot EV activity (only if simulate fleet separately)
    # offers = simulate_fleet(config.NUM_EVS, start_ts, config.SIMULATION_DAYS)
    # plot_ev_charging_activity(offers, start_ts, config.SIMULATION_DAYS)

    print("Evaluation finished!")

