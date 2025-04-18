from optimization.flexOfferOptimizer import optimize, mFRR_reserve_only, optimize_Spot_and_mfrr, sequential_schedule_mfrr_then_spot, optimize_spot_reserve_activation
from classes.electricVehicle import ElectricVehicle
from database.dataManager import get_prices_in_range
import pulp
from datetime import datetime
import pytest
from config import config
import pandas as pd
from flexoffer_logic import Flexoffer, TimeSlice

def test_optimizer():
    Amount = 1000
    fos = []

    print(config.TIME_RESOLUTION)


    for i in range(Amount):
        ev = ElectricVehicle(
            vehicle_id=i,
            capacity=100, #kwh
            soc_min=0.7, #%
            soc_max=0.9, #%
            charging_power=7.0,
            charging_efficiency=0.95
        )
        fos.append(ev.create_flex_offer(tec_fo=True))

       # Run joint (combined) optimization
    joint_fos, joint_results = optimize_Spot_and_mfrr(fos)
    
    # Run sequential scheduler
    seq_fos, seq_results = sequential_schedule_mfrr_then_spot(fos)
    
    # Assemble a DataFrame for comparison
    data = {
        "Strategy": ["Joint", "Sequential"],
        "Revenue_DKK": [joint_results["revenue"], seq_results["revenue"]],
        "Runtime_s": [joint_results["runtime"], seq_results["runtime"]]
    }
    df = pd.DataFrame(data)
    
    # Save comparison to CSV
    df.to_csv("scheduler_comparison.csv", index=False)