import pulp
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import datetime, timedelta
from typing import List
from database.dataManager import get_price_at_datetime, get_prices_in_range, fetch_mFRR_by_date, fetch_mFRR_by_range
import pandas as pd
import pulp
from config import config
from flexoffer_logic import Flexoffer, TimeSlice

def optimize_dual_market(FO: Flexoffer) -> Flexoffer:
    logger.info("Starting dual-market optimization...")
    

    time_horizon = abs(FO.get_lst_hour() - FO.get_et_hour()) #int
    spot_prices = get_prices_in_range(FO.get_est(), FO.get_et()) #List[float]

    time_slots = get_time_slots()
    D_lb, D_ub = get_aggregated_bounds()

    p_spot = get_spot_prices()
    
    mfrr_prices = get_mfrr_prices()  
    r_up   = mfrr_prices.get('r_up', {})
    r_down = mfrr_prices.get('r_down', {})
    
    # Create the PuLP optimization problem (Maximization)
    prob = pulp.LpProblem("DualMarketOptimization", pulp.LpMaximize)
    
    D_sch = {t: pulp.LpVariable(f"D_sch_{t}", lowBound=D_lb[t], upBound=D_ub[t])
             for t in time_slots}
    
    v_spot = {t: pulp.LpVariable(f"v_spot_{t}", lowBound=0)
              for t in time_slots}
    
    # Variables to linearize the absolute imbalance: 
    # v_spot[t] - D_sch[t] = delta_plus[t] - delta_minus[t]
    delta_plus = {t: pulp.LpVariable(f"delta_plus_{t}", lowBound=0)
                  for t in time_slots}
    delta_minus = {t: pulp.LpVariable(f"delta_minus_{t}", lowBound=0)
                   for t in time_slots}
    
    for t in time_slots:
        prob += v_spot[t] - D_sch[t] == delta_plus[t] - delta_minus[t], f"imbalance_constraint_{t}"
    
    up_reg = {t: (D_sch[t] - D_lb[t]) / slot_duration for t in time_slots}
    down_reg = {t: (D_ub[t] - D_sch[t]) / slot_duration for t in time_slots}
    
    # Regulation revenue: calculated per kW.
    # Revenue per time slot = r_up * up_reg + r_down * down_reg.
    reg_revenue = {
        t: r_up[t] * up_reg[t] + r_down[t] * down_reg[t]
        for t in time_slots
    }
    
    
    # Objective: maximize total profit = regulation revenue - (spot cost + imbalance cost)
    prob += pulp.lpSum([
        reg_revenue[t] - p_spot[t] * v_spot[t]
        for t in time_slots
    ]), "TotalProfit"
    
    # Solve the optimization problem
    logger.info("Solving optimization problem using CBC solver...")
    solver = pulp.PULP_CBC_CMD(msg=True)
    result_status = prob.solve(solver)
    logger.info("Optimization finished with status: %s", pulp.LpStatus[prob.status])
    
    # Prepare results dictionary
    results = {}
    for t in time_slots:
        D_sch_val = D_sch[t].varValue
        v_spot_val = v_spot[t].varValue
        imbalance_val = delta_plus[t].varValue + delta_minus[t].varValue
        # Compute power regulation capacities in kW
        up_reg_val = (D_sch_val - D_lb[t]) / slot_duration
        down_reg_val = (D_ub[t] - D_sch_val) / slot_duration
        results[t] = {
            "D_sch": D_sch_val,         # kWh
            "v_spot": v_spot_val,       # kWh
            "up_reg": up_reg_val,       # kW
            "down_reg": down_reg_val    # kW
        }
        logger.info("Time Slot %s: %s", t, results[t])
    
    return results

# ----------------------------
# Module Test/Execution
# ----------------------------
if __name__ == "__main__":
    # For standalone testing, assume each time slot is 1 hour.
    results = optimize_dual_market(slot_duration=1.0)
    print("Dual-Market Optimization Results:")
    for t, res in sorted(results.items()):
        print(f"\nTime Slot {t}:")
        print(f"  Scheduled Baseline (D_sch): {res['D_sch']} kWh")
        print(f"  Spot Purchase (v_spot): {res['v_spot']} kWh")
        print(f"  Imbalance: {res['imbalance']} kWh")
        print(f"  Upward Regulation Capacity: {res['up_reg']} kW")
        print(f"  Downward Regulation Capacity: {res['down_reg']} kW")
