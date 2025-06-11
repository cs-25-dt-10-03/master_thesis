# Spot Market Module with DFO support
import pulp
from flexoffer_logic import Flexoffer, DFO

class SpotMarket:
    def __init__(self, spot_prices):
        self.spot_prices = spot_prices

    def create_variables(self, model):
        for a, offer in enumerate(model.offers):
            dur = offer.get_duration()
            for j in range(dur):
                t = model.offsets[a] + j
                if t < 0 or t >= model.T:
                    continue

                if isinstance(offer, Flexoffer):
                    ts = offer.get_profile()[j]
                    var = pulp.LpVariable(f"p_{a}_{t}", lowBound=ts.min_power, upBound=ts.max_power)
                else:
                    var = pulp.LpVariable(f"p_{a}_{t}", lowBound=0)
                model.p[(a, t)] = var

    def add_constraints(self, model):
        for a, offer in enumerate(model.offers):
            if isinstance(offer, Flexoffer):
                total_energy = pulp.lpSum(model.p[(a, model.offsets[a]+j)] * model.dt
                                          for j in range(offer.get_duration())
                                          if (a, model.offsets[a]+j) in model.p)
                model.prob += total_energy >= offer.get_min_overall_alloc(), f"total_min_energy_{a}"
                model.prob += total_energy <= offer.get_max_overall_alloc(), f"total_max_energy_{a}"

            else:
                poly_list = offer.polygons
                energies = []
                for j, poly in enumerate(poly_list):
                    t = model.offsets[a] + j
                    if (a, t) not in model.p:
                        continue
                    y_vals = [pt.y for pt in poly.points]
                    y_min, y_max = min(y_vals), max(y_vals)
                    model.prob += model.p[(a, t)] >= y_min, f"dfo_min_{a}_{t}"
                    model.prob += model.p[(a, t)] <= y_max, f"dfo_max_{a}_{t}"

                    energies.append(model.p[(a, t)] * model.dt)

                # now *correctly* enforce overall energy
                model.prob += pulp.lpSum(energies) >= offer.min_total_energy, f"dfo_total_min_{a}"
                model.prob += pulp.lpSum(energies) <= offer.max_total_energy, f"dfo_total_max_{a}"

    def build_objective(self, model):
        dt = model.dt
        model.objective_terms.extend(
            -self.spot_prices.iloc[t] * model.p[(a, t)] * dt
            for (a, t) in model.p
        )
