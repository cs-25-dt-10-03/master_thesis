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
                if t >= model.T:
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
                cumulative = pulp.LpAffineExpression()
                poly_list = offer.polygons
                for j, poly in enumerate(poly_list):
                    t = model.offsets[a] + j
                    if (a, t) not in model.p:
                        continue

                    points = poly.points
                    if len(points) < 4:
                        ymin, ymax = points[0].y, points[1].y
                        model.prob += model.p[(a, t)] >= ymin, f"dfo_min_{a}_{t}"
                        model.prob += model.p[(a, t)] <= ymax, f"dfo_max_{a}_{t}"
                    else:
                        for k in range(1, len(points) - 2, 2):
                            x0, y0 = points[k - 1].x, points[k - 1].y
                            x1, y1 = points[k + 1].x, points[k + 1].y
                            if x1 != x0:
                                slope = (y1 - y0) / (x1 - x0)
                                energy_min = y0 + slope * (cumulative - x0)
                                model.prob += model.p[(a, t)] >= energy_min, f"dfo_min_slope_{a}_{t}"
                                break

                    cumulative += model.p[(a, t)] * model.dt

                total_energy = pulp.lpSum(model.p[(a, t)] * model.dt for j, poly in enumerate(poly_list)
                                          if (a, model.offsets[a]+j) in model.p)
                model.prob += total_energy >= offer.min_total_energy, f"dfo_total_min_{a}"
                model.prob += total_energy <= offer.max_total_energy, f"dfo_total_max_{a}"

    def build_objective(self, model):
        dt = model.dt
        model.objective_terms.extend(
            -self.spot_prices.iloc[t] * model.p[(a, t)] * dt
            for (a, t) in model.p
        )
