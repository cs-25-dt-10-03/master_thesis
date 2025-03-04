import pytest
from classes.electricVehicle import ElectricVehicle
from optimization.flexOfferOptimizer import FlexOfferOptimizer
from config import config
from datetime import timedelta
from database.SpotPriceData import SpotPriceData

@pytest.fixture
def ev():
    return ElectricVehicle(
        vehicle_id=1,
        capacity=100,
        soc_min=0.2,
        soc_max=0.8,
        charging_power=7.0,
        charging_efficiency=0.95
    )

def optimizer(ev):
    flex_offers = []
    fo = ev.create_flex_offer(tec_fo=True)
    flex_offers.append(fo)

    FlexOfferOptimizer(flex_offers = flex_offers, time_resolution = timedelta(minutes = 60))
    fos = FlexOfferOptimizer.optimize()
    print(fos[0])
    assert fos[0].scheduled_start is not None