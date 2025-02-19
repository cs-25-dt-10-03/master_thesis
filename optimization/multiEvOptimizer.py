from docplex.mp.model import Model
from datetime import timedelta
from optimization.BaseOptimizer import BaseOptimizer
from typing import Optional

class MultiEvOptimizer(BaseOptimizer):
    def build_model(self):

        self.model = Model("MultiFlexOfferOptimization")
        time_res = timedelta(minutes=self.config.TIME_RESOLUTION)
        offers = self.flex_offers

        # Dictionaries to store per-flex_offer data.
        self.candidate_starts = {}  # {a: [candidate start times]} (optimization vil finde den bedste start tid)
        self.num_candidates = {}    # {a: number of candidate starts}
        self.num_slots = {}         # {a: number of timeslots}

        # Create candidate start times for each offer.
        for a, offer in enumerate(offers):
            cand = []
            t = offer.earliest_start
            while t <= offer.latest_start:
                cand.append(t)
                t += time_res
            self.candidate_starts[a] = cand
            self.num_candidates[a] = len(cand)
            self.num_slots[a] = len(offer.energy_profile)

        # Decision variables:
        # y[(a,j)]: binary variable for flexOffer a, candidate j.
        # x[(a,i)]: continuous energy allocation for flexOffer a, timeslot i.
        # z[(a,i,j)]: auxiliary variable for linearization.
        y = {}
        x = {}
        z = {}
        M = {}  # Upper bounds for x (from each offer's energy_profile).

        for a, offer in enumerate(offers):
            # Create binary variables for candidate start selection.
            for j in range(self.num_candidates[a]):
                y[(a, j)] = self.model.binary_var(name=f"y_{a}_{j}")
            # Create continuous variables for energy allocation.
            for i, (lb, ub) in enumerate(offer.energy_profile):
                x[(a, i)] = self.model.continuous_var(lb=lb, ub=ub, name=f"x_{a}_{i}")
                M[(a, i)] = ub  # Use the upper bound as M.
            # Create auxiliary variables for linking.
            for i in range(self.num_slots[a]):
                for j in range(self.num_candidates[a]):
                    z[(a, i, j)] = self.model.continuous_var(lb=0, ub=M[(a, i)], name=f"z_{a}_{i}_{j}")

        # Constraints for each flexOffer.
        for a, offer in enumerate(offers):
            # Ensure exactly one candidate start is selected.
            self.model.add_constraint(
                self.model.sum(y[(a, j)] for j in range(self.num_candidates[a])) == 1,
                f"OneStartTime_{a}"
            )
            # (Optional) Total energy constraint
            if offer.max_energy is not None:
                total_energy = self.model.sum(x[(a, i)] for i in range(self.num_slots[a]))

                # Ensure min_energy <= total_energy
                if offer.min_energy is not None:
                    self.model.add_constraint(total_energy >= offer.min_energy, f"MinTotalEnergy_{a}")

                # Ensure total_energy <= max_energy
                self.model.add_constraint(total_energy <= offer.max_energy, f"MaxTotalEnergy_{a}")
            # Linking constraints for z: enforce z[a,i,j] = x[a,i]*y[a,j] using Big-M.
            for i in range(self.num_slots[a]):
                for j in range(self.num_candidates[a]):
                    self.model.add_constraint(z[(a, i, j)] <= x[(a, i)], f"Link1_{a}_{i}_{j}")
                    self.model.add_constraint(z[(a, i, j)] <= M[(a, i)] * y[(a, j)], f"Link2_{a}_{i}_{j}")
                    self.model.add_constraint(
                        z[(a, i, j)] >= x[(a, i)] - M[(a, i)] * (1 - y[(a, j)]),
                        f"Link3_{a}_{i}_{j}"
                    )

        # Objective: sum the cost over all offers.
        total_cost_expr = 0
        for a, offer in enumerate(offers):
            for j in range(self.num_candidates[a]):
                for i in range(self.num_slots[a]):
                    candidate_start = self.candidate_starts[a][j]
                    price = self.spot_price_data.get_price_by_datetime(candidate_start + i * time_res)
                    total_cost_expr += price * z[(a, i, j)]
        self.model.minimize(total_cost_expr)

    def solve(self):
        solution = self.model.solve()
        if solution is None:
            return None

        self.optimal_solution = {}  # {a: {"start": chosen_start, "allocation": [x values]}}
        for a, offer in enumerate(self.flex_offers):
            chosen_candidate = None
            for j in range(self.num_candidates[a]):
                var = self.model.get_var_by_name(f"y_{a}_{j}")
                if solution.get_value(var) > 0.5:
                    chosen_candidate = self.candidate_starts[a][j]
                    break
            allocation = [
                solution.get_value(self.model.get_var_by_name(f"x_{a}_{i}"))
                for i in range(self.num_slots[a])
            ]
            self.optimal_solution[a] = {"start": chosen_candidate, "allocation": allocation}
        return self.optimal_solution

    def update_schedule(self):
        """
        Update each flexOffer with its computed optimal start time and energy allocation.
        """
        for a, offer in enumerate(self.flex_offers):
            sol = self.optimal_solution.get(a, None)
            if sol is not None:
                offer.scheduled_start = sol["start"]
                offer.scheduled_energy_profile = sol["allocation"]
