from pulp import LpProblem, LpMinimize, LpVariable, lpSum
from datetime import timedelta
import pandas as pd
from config import config

class FlexOfferOptimizer:
    def __init__(self, flex_offers, spot_price_data, time_resolution=config.TIME_RESOLUTION):
        self.flex_offers = flex_offers
        self.spot_price_data = spot_price_data
        self.time_resolution = timedelta(minutes=time_resolution)

    def optimize(self):
        prob = LpProblem("FlexOfferScheduling", LpMinimize)

        # Decision variables:
        # - For each offer, a binary variable for each possible start time.
        # - For each offer and each possible start time, a continuous energy variable for each time slot.
        start_vars = {}
        energy_vars = {}

        for offer in self.flex_offers:
            # Define possible start times within the allowed window.
            start_times = pd.date_range(start=offer.earliest_start, end=offer.latest_start, freq=self.time_resolution)
            start_vars[offer] = LpVariable.dicts(f"Start_{offer.offer_id}", start_times, cat='Binary')
            energy_vars[offer] = {}
            for t in start_times:
                energy_vars[offer][t] = []
                for i, (min_e, max_e) in enumerate(offer.energy_profile):
                    var = LpVariable(f"Energy_{offer.offer_id}_{t.strftime('%Y%m%d%H%M')}_{i}", lowBound=0, cat='Continuous')
                    energy_vars[offer][t].append(var)

        # Objective: minimize total cost across all offers.
        total_cost = []
        for offer in self.flex_offers:
            for t in start_vars[offer]:
                for i, (min_e, max_e) in enumerate(offer.energy_profile):
                    # The spot price is taken at the time slot corresponding to the start time + offset.
                    price = self.spot_price_data.get_price_by_datetime(t + i * self.time_resolution)
                    total_cost.append(price * energy_vars[offer][t][i])
        prob += lpSum(total_cost)

        # Constraints:
        for offer in self.flex_offers:
            # (1) Exactly one start time must be selected.
            prob += lpSum(start_vars[offer][t] for t in start_vars[offer]) == 1, f"OneStart_{offer.offer_id}"

            # (2) For each potential start time and slot, link the energy variable with the start decision.
            for t in start_vars[offer]:
                for i, (min_e, max_e) in enumerate(offer.energy_profile):
                    # If start time t is not chosen (x = 0), then the energy must be 0.
                    # If chosen (x = 1), energy must lie between min_e and max_e.
                    prob += energy_vars[offer][t][i] <= max_e * start_vars[offer][t], f"MaxEnergy_{offer.offer_id}_{t}_{i}"
                    prob += energy_vars[offer][t][i] >= min_e * start_vars[offer][t], f"MinEnergy_{offer.offer_id}_{t}_{i}"

            # (3) Optional overall energy constraints (if provided for the flexOffer).
            if offer.min_energy is not None:
                total_energy = lpSum(energy_vars[offer][t][i]
                                     for t in start_vars[offer]
                                     for i in range(len(offer.energy_profile)))
                prob += total_energy >= offer.min_energy, f"TotalMinEnergy_{offer.offer_id}"
                
            if offer.max_energy is not None:
                total_energy = lpSum(energy_vars[offer][t][i]
                                     for t in start_vars[offer]
                                     for i in range(len(offer.energy_profile)))
                prob += total_energy <= offer.max_energy, f"TotalMaxEnergy_{offer.offer_id}"

        # Solve the MILP
        prob.solve()

        # Extract and set scheduled start time and energy profile from the chosen variables.
        for offer in self.flex_offers:
            chosen_start = None
            for t in start_vars[offer]:
                if start_vars[offer][t].varValue == 1:
                    chosen_start = t
                    break
            if chosen_start is not None:
                offer.scheduled_start = chosen_start
                # The scheduled energy profile is taken from the energy variables corresponding to the chosen start.
                offer.scheduled_energy_profile = [energy_vars[offer][chosen_start][i].varValue
                                                  for i in range(len(offer.energy_profile))]
        return self.flex_offers
