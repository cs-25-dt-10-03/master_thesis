from classes.electricVehicle import ElectricVehicle
from database.dataManager import fetch_all_offers
from aggregation.alignments import startAlignment

def main():
    tesla_model_y = ElectricVehicle(
            vehicle_id=1,
            capacity=75.0,
            soc_min=0.20,
            soc_max=0.80,
            charging_power=7.0,
            charging_efficiency=0.84,
            initial_soc=0.30
            )

    tesla_model_s = ElectricVehicle(
            vehicle_id=2,
            capacity=100.0,
            soc_min=0.40,
            soc_max=0.80,
            charging_power=10.0,
            charging_efficiency=0.84,
            initial_soc=0.30
            )
    fo1 = tesla_model_y.create_flex_offer()
    fo2 = tesla_model_s.create_flex_offer()

    print(startAlignment([fo1,fo2]))

    result = fetch_all_offers()

"""
    for sfo in result["sfo"]:
        print(sfo)

    for tec in result["tec"]:
        print(tec)
"""
    

if __name__ == "__main__":
    main()
