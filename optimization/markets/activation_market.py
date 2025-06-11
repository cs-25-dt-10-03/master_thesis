# optimization/markets/activation_market.py

import pulp
from flexoffer_logic import Flexoffer, DFO

class ActivationMarket:
    def __init__(self, activation_prices, indicators):
        # activation_prices: DataFrame with columns
        #   'BalancingPowerPriceUpDKK', 'BalancingPowerPriceDownDKK',
        #   'ImbalancePriceDKK', plus alpha_up/alpha_dn in indicators
        self.activation_prices = activation_prices
        self.indicators       = indicators

    def create_variables(self, model):
        # For every cleared reserve slot, allow dispatch + slack
        for (a, t) in model.pr_up:
            model.pb_up[(a, t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
            model.pb_dn[(a, t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
            model.s_up[(a, t)]  = pulp.LpVariable(f"s_up_{a}_{t}",  lowBound=0)
            model.s_dn[(a, t)]  = pulp.LpVariable(f"s_dn_{a}_{t}",  lowBound=0)

    def add_constraints(self, model):
        for (a, t) in model.pr_up:
            α_up, α_dn = self.indicators[t]
            pr_up_var  = model.pr_up[(a, t)]
            pr_dn_var  = model.pr_dn[(a, t)]
            pb_up_var  = model.pb_up[(a, t)]
            pb_dn_var  = model.pb_dn[(a, t)]
            s_up_var   = model.s_up[(a, t)]
            s_dn_var   = model.s_dn[(a, t)]
            p_at       = model.p[(a, t)]

            # 1) Cannot call more than reserved
            model.prob += pb_up_var <= pr_up_var, f"act_call_up_limit_{a}_{t}"
            model.prob += pb_dn_var <= pr_dn_var, f"act_call_dn_limit_{a}_{t}"
            # 2) Fractional caps
            model.prob += pb_up_var <= α_up * pr_up_var, f"act_frac_up_{a}_{t}"
            model.prob += pb_dn_var <= α_dn * pr_dn_var, f"act_frac_dn_{a}_{t}"
            # 3) Slack if under‐called
            model.prob += pb_up_var + s_up_var >= α_up * pr_up_var, f"act_slack_up_{a}_{t}"
            model.prob += pb_dn_var + s_dn_var >= α_dn * pr_dn_var, f"act_slack_dn_{a}_{t}"

            # 4) Enforce post‐call power within offer bounds
            p_act = p_at - pb_up_var + pb_dn_var
            offer = model.offers[a]
            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += p_act >= ts.min_power, f"act_min_power_{a}_{t}"
                model.prob += p_act <= ts.max_power, f"act_max_power_{a}_{t}"
            else:  # DFO
                poly   = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                y_min, y_max = min(y_vals), max(y_vals)
                model.prob += p_act >= y_min, f"dfo_act_min_{a}_{t}"
                model.prob += p_act <= y_max, f"dfo_act_max_{a}_{t}"

            # 5) Total delivered ≥ minimum overall allocation
            #    (post‐activation)
            duration = (offer.get_duration() 
                        if isinstance(offer, Flexoffer) 
                        else offer.get_duration())
            expr = pulp.lpSum(
                ( model.p[(a, tt)]
                  - model.pb_up[(a, tt)]
                  + model.pb_dn[(a, tt)]
                ) * model.dt
                for tt in range(model.offsets[a],
                                model.offsets[a] + duration)
                if (a, tt) in model.p
            )
            min_alloc = (offer.get_min_overall_alloc() 
                         if isinstance(offer, Flexoffer) 
                         else offer.min_total_energy)
            model.prob += expr >= min_alloc, f"act_total_min_{a}_{t}"

    def build_objective(self, model):
        dt = model.dt
        for (a, t) in model.pb_up:
            row      = self.activation_prices.iloc[t]
            b_up     = row['BalancingPowerPriceUpDKK']
            b_dn     = row['BalancingPowerPriceDownDKK']
            penalty  = row['ImbalancePriceDKK']
            spot_px  = model.spot_prices.iloc[t]
            α_up, α_dn = self.indicators[t]

            # Only reward if there was nonzero activation fraction
            rev_up  = (b_up  - spot_px) * model.pb_up[(a, t)] * dt if α_up > 0 else 0
            rev_dn  = (b_dn  - spot_px) * model.pb_dn[(a, t)] * dt if α_dn > 0 else 0
            cost_slack = -penalty * (model.s_up[(a, t)] + model.s_dn[(a, t)]) * dt

            model.objective_terms.extend([rev_up, rev_dn, cost_slack])
