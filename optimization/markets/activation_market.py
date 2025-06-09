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
            # Pull activation fraction indicators
            α_up, α_dn = self.indicators[t]

            # 1) Bid limits
            model.prob += model.pb_up[(a, t)] <= model.pr_up[(a, t)], \
                          f"act_pb_up_limit_{a}_{t}"
            model.prob += model.pb_dn[(a, t)] <= model.pr_dn[(a, t)], \
                          f"act_pb_dn_limit_{a}_{t}"

            # 2) Fractional up/down call caps
            model.prob += model.pb_up[(a, t)] <= α_up * model.pr_up[(a, t)], \
                          f"act_pb_up_fraction_{a}_{t}"
            model.prob += model.pb_dn[(a, t)] <= α_dn * model.pr_dn[(a, t)], \
                          f"act_pb_dn_fraction_{a}_{t}"

            # 3) Slack penalties if short‐called
            model.prob += (model.pb_up[(a, t)] + model.s_up[(a, t)]
                           >= α_up * model.pr_up[(a, t)]), f"act_up_slack_{a}_{t}"
            model.prob += (model.pb_dn[(a, t)] + model.s_dn[(a, t)]
                           >= α_dn * model.pr_dn[(a, t)]), f"act_dn_slack_{a}_{t}"

            # 4) Enforce p_act = p – pb_up + pb_dn within min/max power
            offer = model.offers[a]
            p_act = model.p[(a, t)] - model.pb_up[(a, t)] + model.pb_dn[(a, t)]

            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += p_act >= ts.min_power, \
                              f"act_min_power_{a}_{t}"
                model.prob += p_act <= ts.max_power, \
                              f"act_max_power_{a}_{t}"
            else:
                poly   = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                y_min, y_max = min(y_vals), max(y_vals)
                model.prob += p_act >= y_min, \
                              f"dfo_act_min_{a}_{t}"
                model.prob += p_act <= y_max, \
                              f"dfo_act_max_{a}_{t}"

            # 5) (Optional) system‐level caps, if desired
            # for t_index in sorted({tt for (_, tt) in model.pr_up}):
            #     model.prob += (
            #         pulp.lpSum(model.pb_up[(i, t_index)]
            #                    for (i, tt) in model.pr_up
            #                    if tt == t_index)
            #         <= self.activation_prices["mFRRUpActBal"].iloc[t_index],
            #         f"act_total_up_cap_{t_index}"
            #     )
            #     model.prob += (
            #         pulp.lpSum(model.pb_dn[(i, t_index)]
            #                    for (i, tt) in model.pr_dn
            #                    if tt == t_index)
            #         <= self.activation_prices["mFRRDownActBal"].iloc[t_index],
            #         f"act_total_dn_cap_{t_index}"
            #     )

            # 6) Enforce total energy delivered ≥ min_overall_alloc
            #    after all up/down dispatches
            p_act_total = pulp.lpSum(
                (model.p[(a, tt)]
                   - model.pb_up[(a, tt)]
                   + model.pb_dn[(a, tt)]) * model.dt
                for tt in range(model.offsets[a],
                                model.offsets[a] + offer.get_duration())
                if (a, tt) in model.p
            )

            if isinstance(offer, Flexoffer):
                model.prob += (
                    p_act_total >= offer.get_min_overall_alloc(),
                    f"post_activation_energy_{a}_{t}"
                )


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
