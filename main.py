from database.dataManager import fetchAveragePower
from datetime import datetime, timedelta
from classes.electricVehicle import ElectricVehicle
from config import config
from classes.DFO import DFO, DependencyPolygon
from aggregation.DFO_aggregation import agg2to1

def main():
    tesla_model_y = ElectricVehicle(
        vehicle_id="TeslaModelY_1",
        capacity=75.0,
        soc_min=0.20,
        soc_max=0.80,
        charging_power=7.0,
        charging_efficiency=0.84,
        initial_soc=0.30
    )

    tesla_model_s = ElectricVehicle(
        vehicle_id="TeslaModelS_1",
        capacity=100.0,
        soc_min=0.40,
        soc_max=0.80,
        charging_power=10.0,
        charging_efficiency=0.84,
        initial_soc=0.30
    )

    charging_window_start = datetime.now().replace(hour=22, minute=0, second=0, microsecond=0)
    charging_window_end = charging_window_start + timedelta(hours=8)
    duration = timedelta(hours=3) 

    dfo1 = tesla_model_y.create_dfo(charging_window_start,
                                   charging_window_end,
                                   duration,
                                   numsamples=4)

    dfo2 = tesla_model_s.create_dfo(charging_window_start,
                                   charging_window_end,
                                   duration,
                                   numsamples=4)
    
    print("Generated DFO for", tesla_model_y.vehicle_id)
    #dfo.print_dfo()
    #print(dfo)
    dfo1.plot_dfo()
    dfo2.plot_dfo()
    dfo3 = agg2to1(dfo1, dfo2, 4)
    dfo3.plot_dfo()

if __name__ == "__main__":
    main()
