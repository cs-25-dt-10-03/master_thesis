# optimization/markets/activation_market.py

import pulp
from flexoffer_logic import Flexoffer, DFO

class ActivationMarket:
    def __init__(self, activation_prices, indicators):
        # activation_prices: DataFrame with columns
        #   'BalancingPowerPriceUpDKK', 'BalancingPowerPriceDownDKK',
        #   'ImbalancePriceDKK'
        # indicators:      list or DataFrame of (α_up, α_dn) per period
        self.activation_prices = activation_prices
        self.indicators       = indicators

    def create_variables(self, model):
        # For every cleared reserve slot, allow dispatch + slack
        for (a, t) in model.pr_up:
            model.pb_up[(a, t)] = pulp.LpVariable(f"pb_up_{a}_{t}", lowBound=0)
            model.pb_dn[(a, t)] = pulp.LpVariable(f"pb_dn_{a}_{t}", lowBound=0)
            # Slack for under‐delivery
            model.s_up[(a, t)] = pulp.LpVariable(f"s_up_{a}_{t}", lowBound=0)
            model.s_dn[(a, t)] = pulp.LpVariable(f"s_dn_{a}_{t}", lowBound=0)

    def add_constraints(self, model):
        dt = model.dt     # e.g. 0.25 h for 15 min
        # Energinet mFRR FAT spec
        PREP = 2.5        # minutes
        RAMP = 10.0       # minutes
        STEP = dt * 60.0  # length of our period in minutes

        # compute the average‐power shape factor over one period
        M = min(PREP + RAMP, STEP)
        area1 = max(0.0, (M - PREP)**2 / (2.0 * RAMP))
        area2 = max(0.0, STEP - (PREP + RAMP))
        shape_factor = (area1 + area2) / STEP  # ≈ 0.5 for 15 min, 2.5+10 ramp

        for (a, t), pb_up_var in list(model.pb_up.items()):
            pb_dn_var = model.pb_dn[(a, t)]
            s_up_var  = model.s_up[(a, t)]
            s_dn_var  = model.s_dn[(a, t)]
            pr_up_var = model.pr_up[(a, t)]
            pr_dn_var = model.pr_dn[(a, t)]
            α_up, α_dn = self.indicators[t]

            # 1) Cannot dispatch unless there's a request
            model.prob += pb_up_var <= α_up * pr_up_var, f"act_req_up_{a}_{t}"
            model.prob += pb_dn_var <= α_dn * pr_dn_var, f"act_req_dn_{a}_{t}"

            # 2) Fat‐shaped caps: only shape_factor·capacity is deliverable
            model.prob += (
                pb_up_var <= α_up * pr_up_var * shape_factor,
                f"act_shape_up_{a}_{t}"
            )
            model.prob += (
                pb_dn_var <= α_dn * pr_dn_var * shape_factor,
                f"act_shape_dn_{a}_{t}"
            )

            # 3) Slack if under‐called
            model.prob += (
                pb_up_var + s_up_var >= α_up * pr_up_var * shape_factor,
                f"act_slack_up_{a}_{t}"
            )
            model.prob += (
                pb_dn_var + s_dn_var >= α_dn * pr_dn_var * shape_factor,
                f"act_slack_dn_{a}_{t}"
            )

            # 4) Enforce post‐call power within the original offer bounds
            p_nom     = model.p[(a, t)]
            p_act     = p_nom - pb_up_var + pb_dn_var
            offer     = model.offers[a]

            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += p_act >= ts.min_power, f"act_min_power_{a}_{t}"
                model.prob += p_act <= ts.max_power, f"act_max_power_{a}_{t}"
            else:  # DFO
                df = offer.get_profile()[t - model.offsets[a]]
                model.prob += p_act >= df.min_down, f"act_min_power_dfo_{a}_{t}"
                model.prob += p_act <= df.max_up,   f"act_max_power_dfo_{a}_{t}"

    def build_objective(self, model):
        dt = model.dt
        for (a, t), pb_up_var in model.pb_up.items():
            pb_dn_var = model.pb_dn[(a, t)]
            s_up_var  = model.s_up[(a, t)]
            s_dn_var  = model.s_dn[(a, t)]

            row     = self.activation_prices.iloc[t]
            b_up    = row['BalancingPowerPriceUpDKK']
            b_dn    = row['BalancingPowerPriceDownDKK']
            pen     = row['ImbalancePriceDKK']
            spot_px = model.spot_prices.iloc[t]
            α_up, α_dn = self.indicators[t]

            # reward only when there's a positive request
            rev_up = (b_up - spot_px) * pb_up_var * dt if α_up > 0 else 0
            rev_dn = (b_dn - spot_px) * pb_dn_var * dt if α_dn > 0 else 0
            cost_slack = -pen * (s_up_var + s_dn_var) * dt

            model.objective_terms.extend([rev_up, rev_dn, cost_slack])
