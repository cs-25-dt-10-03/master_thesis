# optimization/markets/reserve_market.py

import pulp
from flexoffer_logic import Flexoffer, DFO

class ReserveMarket:
    def __init__(self, reserve_prices):
        # reserve_prices: DataFrame with columns
        #   'mFRR_UpPriceDKK', 'mFRR_DownPriceDKK',
        #   'mFRR_UpPurchased', 'mFRR_DownPurchased'
        self.reserve_prices = reserve_prices

    def create_variables(self, model):
        # Create up/down capacity variables for every p‐slot
        for (a, t) in model.p:
            model.pr_up[(a, t)] = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=0)
            model.pr_dn[(a, t)] = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=0)

    def add_constraints(self, model):
        # 1) Per‐offer headroom constraints
        for (a, t) in model.p:
            offer = model.offers[a]
            p_at = model.p[(a, t)]

            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                min_feasible = ts.min_power
                max_feasible = ts.max_power
            else:  # DFO
                poly = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                min_feasible, max_feasible = min(y_vals), max(y_vals)

            # Up‐reserve ≤ (baseline − minimum)
            model.prob += (
                model.pr_up[(a, t)] <= p_at - min_feasible,
                f"res_up_bound_{a}_{t}"
            )
            # Down‐reserve ≤ (maximum − baseline)
            model.prob += (
                model.pr_dn[(a, t)] <= max_feasible - p_at,
                f"res_dn_bound_{a}_{t}"
            )

        # 2) System‐level caps
        for t in sorted({t for (_, t) in model.pr_up}):
            purchased_up = self.reserve_prices['mFRR_UpPurchased'].iloc[t]
            purchased_dn = self.reserve_prices['mFRR_DownPurchased'].iloc[t]

            # ∑_a pr_up[a,t] ≤ total procured up‐reserve
            model.prob += (
                pulp.lpSum(model.pr_up[(a, t)] 
                           for (a, tt) in model.pr_up if tt == t)
                <= purchased_up,
                f"sys_cap_res_up_{t}"
            )
            # ∑_a pr_dn[a,t] ≤ total procured down‐reserve
            model.prob += (
                pulp.lpSum(model.pr_dn[(a, t)] 
                           for (a, tt) in model.pr_dn if tt == t)
                <= purchased_dn,
                f"sys_cap_res_dn_{t}"
            )

    def build_objective(self, model):
        dt = model.dt
        # Harvest only positive clearing prices
        for (a, t) in model.pr_up:
            row = self.reserve_prices.iloc[t]
            r_up  = row['mFRR_UpPriceDKK']
            r_dn  = row['mFRR_DownPriceDKK']

            if r_up > 0:
                model.objective_terms.append(r_up * model.pr_up[(a, t)] * dt)
            else:
                # no revenue ⇒ force zero capacity
                model.prob += (model.pr_up[(a, t)] == 0), f"no_up_when_no_price_{a}_{t}"

            if r_dn > 0:
                model.objective_terms.append(r_dn * model.pr_dn[(a, t)] * dt)
            else:
                model.prob += (model.pr_dn[(a, t)] == 0), f"no_dn_when_no_price_{a}_{t}"
