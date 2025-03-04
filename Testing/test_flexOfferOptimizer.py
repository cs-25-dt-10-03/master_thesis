# import pytest
# from datetime import datetime, timedelta
# from classes.flexOffer import flexOffer
# from optimization.flexOfferOptimizer import FlexOfferOptimizer  # Adjust the import path as needed
# from config import config

# def test_flex_offer_optimizer():
#     t0 = datetime(2024, 1, 1, 8, 0)
#     t1 = datetime(2024, 1, 1, 10, 0)
#     duration = t1 - t0
#     energy_profile = [(0, 7) for _ in range(2)]
#     fo1 = flexOffer(
#         offer_id=1,
#         earliest_start=t0,
#         latest_start=t1,
#         end_time=t1,
#         duration=duration,
#         energy_profile=energy_profile,
#         min_energy=0,
#         max_energy=14
#     )
#     fo2 = flexOffer(
#         offer_id=2,
#         earliest_start=t0,
#         latest_start=t1,
#         end_time=t1,
#         duration=duration,
#         energy_profile=energy_profile,
#         min_energy=0,
#         max_energy=14
#     )
#     optimizer = FlexOfferOptimizer([fo1, fo2], time_resolution=config.TIME_RESOLUTION)
#     optimized_offers = optimizer.optimize()
#     for offer in optimized_offers:
#         assert offer.scheduled_start is not None
#         assert isinstance(offer.scheduled_energy_profile, list)
#         assert len(offer.scheduled_energy_profile) == len(offer.energy_profile)
