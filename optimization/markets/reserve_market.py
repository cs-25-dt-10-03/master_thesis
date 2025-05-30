# Reserve Market Module with DFO support
import pulp
from flexoffer_logic import Flexoffer, DFO

class ReserveMarket:
    def __init__(self, reserve_prices):
        self.reserve_prices = reserve_prices

    def create_variables(self, model):
        for (a, t), _ in model.p.items():
            model.pr_up[(a, t)] = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
            model.pr_dn[(a, t)] = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)

    def add_constraints(self, model):
        for (a, t) in model.p:
            offer = model.offers[a]

            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += model.pr_up[(a, t)] <= ts.max_power, f"r_up_bound_{a}_{t}"
                model.prob += model.pr_dn[(a, t)] <= ts.max_power - model.p[(a, t)], f"r_dn_bound_{a}_{t}"
            else:
                # Use remaining margin above p[(a,t)] for pr_dn, and below for pr_up
                poly = offer.polygons[t - model.offsets[a]]
                points = poly.points
                if len(points) < 4:
                    ymax = points[1].y
                    model.prob += model.pr_up[(a, t)] <= model.p[(a, t)], f"dfo_rup_bound_{a}_{t}"
                    model.prob += model.pr_dn[(a, t)] <= ymax - model.p[(a, t)], f"dfo_rdn_bound_{a}_{t}"

    def build_objective(self, model):
        dt = model.dt
        for (a, t) in model.pr_up:
            r_up, r_dn, *_ = self.reserve_prices.iloc[t]
            model.objective_terms.append(r_up * model.pr_up[(a, t)] * dt)
            model.objective_terms.append(r_dn * model.pr_dn[(a, t)] * dt)
