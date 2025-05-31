# Activation Market Module (DFO agnostic — relies on pr_* constraints)
import pulp
from config import config
from flexoffer_logic import Flexoffer, DFO

class ActivationMarket:
    def __init__(self, activation_prices, indicators):
        self.activation_prices = activation_prices
        self.indicators = indicators

    def create_variables(self, model):
        for (a, t) in model.pr_up:
            model.pb_up[(a, t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
            model.pb_dn[(a, t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
            model.s_up[(a, t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
            model.s_dn[(a, t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    def add_constraints(self, model):
        for (a, t) in model.pr_up:
            d_up, d_dn = self.indicators[t]

            model.prob += model.pb_up[(a, t)] <= model.pr_up[(a, t)], f"act_pb_up_limit_{a}_{t}"
            model.prob += model.pb_dn[(a, t)] <= model.pr_dn[(a, t)], f"act_pb_dn_limit_{a}_{t}"

            if d_up == 0:
                model.prob += model.pb_up[(a, t)] == 0, f"act_pb_up_zero_if_off_{a}_{t}"
            if d_dn == 0:
                model.prob += model.pb_dn[(a, t)] == 0, f"act_pb_dn_zero_if_off_{a}_{t}"

            model.prob += model.pb_up[(a, t)] + model.s_up[(a, t)] >= model.pr_up[(a, t)] * d_up, f"act_up_slack_{a}_{t}"
            model.prob += model.pb_dn[(a, t)] + model.s_dn[(a, t)] >= model.pr_dn[(a, t)] * d_dn, f"act_dn_slack_{a}_{t}"

            # — enforce actual power stays within [min,max] of the time slice —
            #    p_act = p – pb_up + pb_dn
            offer = model.offers[a]
            p_act = ( model.p[(a, t)]
                    - model.pb_up[(a, t)]
                    + model.pb_dn[(a, t)] )

            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += p_act >= ts.min_power, f"act_min_power_{a}_{t}"
                model.prob += p_act <= ts.max_power, f"act_max_power_{a}_{t}"
            else:
                # DFO: bound p_act within global polygon y-limits
                poly   = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                y_min, y_max = min(y_vals), max(y_vals)
                model.prob += p_act >= y_min, f"dfo_act_min_{a}_{t}"
                model.prob += p_act <= y_max, f"dfo_act_max_{a}_{t}"
            # for DFOs you’d analogously bound p_act by the polygon’s y-values

    def build_objective(self, model):
        dt = model.dt
        for (a, t) in model.pb_up:
            b_up, b_dn, penalty, *_ = self.activation_prices.iloc[t]
            d_up, d_dn = self.indicators[t]
            spot = model.spot_prices.iloc[t]

            b_up = b_up if d_up == 1 else 0
            b_dn = b_dn if d_dn == 1 else 0

            model.objective_terms.append((b_up - spot) * model.pb_up[(a, t)] * dt)
            model.objective_terms.append((b_dn - spot) * model.pb_dn[(a, t)] * dt)
            model.objective_terms.append(-penalty * (model.s_up[(a, t)] + model.s_dn[(a, t)]) * dt)
