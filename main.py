from datetime import datetime, timedelta
from classes.electricVehicle import ElectricVehicle
from config import config
from classes.DFO import DFO, DependencyPolygon
from aggregation.DFO_aggregation import agg2to1, aggnto1
from disaggregation.DFO_disaggregation import disagg1to2, disagg1toN
from database.dataManager import fetch_all_offers

def main():
    tesla_model_y = ElectricVehicle(
        vehicle_id=1,
        capacity=75.0,
        soc_min=0.20,
        soc_max=0.80,
        charging_power=7.0,
        charging_efficiency=0.84,
        #initial_soc=0.30
    )

    tesla_model_s = ElectricVehicle(
        vehicle_id=2,
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
        #initial_soc=0.30
    )

    tesla_model_s = ElectricVehicle(
        vehicle_id=2,
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
        #initial_soc=0.30
    )

    charging_window_start = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)
    charging_window_end = charging_window_start + timedelta(hours=8)
    duration = timedelta(hours=3) 

    dfo1 = tesla_model_y.create_dfo(charging_window_start,
                                   charging_window_end,
                                   duration,
                                   numsamples=4)

    dfo2 = tesla_model_s.create_dfo(charging_window_start+timedelta(hours=1),
                                   charging_window_end,
                                   duration,
                                   numsamples=4)
    
    dfo3 = tesla_model_s.create_dfo(charging_window_start+timedelta(hours=2),
                                   charging_window_end,
                                   duration,
                                   numsamples=4)

    #print("Generated DFO for", tesla_model_y.vehicle_id)
    #dfo.print_dfo()
    #print(dfo)
    #dfo1.plot_dfo()
    #dfo2.plot_dfo()
    dfos = [dfo1, dfo2, dfo3]  # Assume these are the original DFOs
    dfo5 = aggnto1(dfos, 4)
    dfo5.plot_dfo()
    # Create sample DFOs

    # Define a reference schedule
    yA_ref = [4.0, 8.0, 10.0, 12.0, 6.0]

    # Disaggregate into two
    #y1_ref, y2_ref = disagg1to2(dfo1, dfo2, dfo3, yA_ref)
    #print("Disaggregated y1_ref:", y1_ref)
    #print("Disaggregated y2_ref:", y2_ref)

    # Disaggregate into multiple DFOs
    y_refs = disagg1toN(dfo5, dfos, yA_ref)
    print("Disaggregated y_refs:", y_refs)
    

if __name__ == "__main__":
    main()
