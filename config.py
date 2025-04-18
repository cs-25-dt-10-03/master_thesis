import json
import os
import flexoffer_logic

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config():
    flexoffer_logic.reload_config()  # Loads config into c++
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


class config:
    _config_data = load_config()

    # Simulations
    NUM_EVS = _config_data.get("NUM_EVS", 1000)
    SIMULATION_DAYS = _config_data.get("SIMULATION_DAYS", 30)
    SIMULATION_START_DATE = _config_data.get("SIMULATION_START_DATE", "2025-03-01")

    # Market requirements
    TIME_RESOLUTION = _config_data.get("TIME_RESOLUTION", 3600)
    MIN_BID_SIZE = _config_data.get("MIN_BID_SIZE", 100)
    REQUIRE_UNIFORM = _config_data.get("REQUIRE_UNIFORM", 0)
    MIN_BID_SIZE = _config_data.get("MIN_BID_SIZE", 1000)

    RESOLUTION = "hourly" if TIME_RESOLUTION == 3600 else "15min"
    DATA_FILEPATH = os.path.join("..", "SmartCharging_2020_to_2032")

    market_config = {
    "modules": {
        "spot": True,
        "reserve": True,
        "activation": True
    },
    "strategy": "joint",  # or "sequential"
}
