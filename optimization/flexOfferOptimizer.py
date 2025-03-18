from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import datetime, timedelta
from typing import List
from database.dataManager import get_price_at_datetime, get_prices_in_range
import pandas as pd
import pulp
from config import config
from flexoffer_logic import Flexoffer, TimeSlice

def optimize(FO: Flexoffer) -> Flexoffer:

    time_horizon = abs(FO.get_lst_hour() - FO.get_et_hour()) #int
    spot_prices = get_prices_in_range(FO.get_est(), FO.get_et()) #List[float]

    #define model
    model = pulp.LpProblem("FlexOffer_Scheduling", pulp.LpMinimize)

    #decision variables
    power_alloc = [pulp.LpVariable(f"power_{t}", lowBound=FO.get_profile()[t].min_power, upBound=FO.get_profile()[t].max_power,
                                   cat="Continuous") for t in range(FO.get_duration())]
    
    start_vars = {s: pulp.LpVariable(f"start_{s}", cat="Binary") for s in FO.get_allowed_start_times()}

    # Objective function
    model += pulp.lpSum(spot_prices[t] * power_alloc[t] for t in range(time_horizon)), "Total_Cost"

    # constraint - only 1 start time
    model += pulp.lpSum(start_vars[s] for s in FO.get_allowed_start_times()) == 1, "Select_One_Start_Time"


    # Constraints: Energy bounds
    for t in range(FO.get_duration()):
        model += power_alloc[t] >= FO.get_profile()[t].min_power
        model += power_alloc[t] <= FO.get_profile()[t].max_power

    # Constraints: Tec total energy
    total_alloc = sum(power_alloc)
    if FO.get_min_overall_alloc() > 0:
        model += total_alloc >= FO.get_min_overall_alloc()
        model += total_alloc <= FO.get_max_overall_alloc()
    
        # Solve the problem
    model.solve()

    # Get the optimal start time and power allocation
    optimal_start_time = None
    for st in FO.get_allowed_start_times():
        if start_vars[st].varValue == 1:
            optimal_start_time = st
            break

    optimal_power_alloc = [p.varValue for p in power_alloc]
    FO.set_scheduled_allocation(optimal_power_alloc)
    FO.set_scheduled_start_time(optimal_start_time)


    return FO