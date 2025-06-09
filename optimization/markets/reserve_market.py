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
                model.prob += model.pr_up[(a, t)] <= model.p[(a, t)], f"r_up_bound_{a}_{t}"
#                model.prob += model.pr_up[(a, t)] <= ts.max_power, f"r_up_bound_{a}_{t}"
                model.prob += model.pr_dn[(a, t)] <= ts.max_power - model.p[(a, t)], f"r_dn_bound_{a}_{t}"
            else:
                # Use remaining margin above p[(a,t)] for pr_dn, and below for pr_up
                poly   = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                y_min, y_max = min(y_vals), max(y_vals)
                # Up-reserve ≤ margin above baseline; down-reserve ≤ margin below max
                model.prob += model.pr_up[(a, t)] <= model.p[(a, t)] - y_min, f"dfo_rup_bound_{a}_{t}"
                model.prob += model.pr_dn[(a, t)] <= y_max - model.p[(a, t)], f"dfo_rdn_bound_{a}_{t}"

            #  → HEADROOM for up‐reserve: baseline p[a,t] must >= pr_up[a,t].
            model.prob += (
                model.p[(a, t)] >= model.pr_up[(a, t)],
                f"headroom_reserve_up_{a}_{t}"
            )

            #  → HEADROOM for down‐reserve: p[a,t] + pr_dn[a,t] <= max_feasible
            if isinstance(offer, Flexoffer):
                ts = offer.get_profile()[t - model.offsets[a]]
                model.prob += (
                    model.p[(a, t)] + model.pr_dn[(a, t)] <= ts.max_power,
                    f"headroom_reserve_dn_{a}_{t}"
                )
            else:
                poly   = offer.polygons[t - model.offsets[a]]
                y_vals = [pt.y for pt in poly.points]
                y_max  = max(y_vals)
                model.prob += (
                    model.p[(a, t)] + model.pr_dn[(a, t)] <= y_max,
                    f"headroom_reserve_dn_dfo_{a}_{t}"
                )
        #  SYSTEM‐LEVEL CAP: ∑_a pr_up[a,t] ≤ total mFRR_UpPurchased[t]
        for t_index in sorted({t for (a, t) in model.pr_up}):
            purchased_up = self.reserve_prices["mFRR_UpPurchased"].iloc[t_index]
            purchased_dn = self.reserve_prices["mFRR_DownPurchased"].iloc[t_index]

            # Cap total up‐reserve by what system actually procured
            model.prob += (
                pulp.lpSum(model.pr_up[(a, t_index)]
                        for (a, tt) in model.pr_up if tt == t_index)
                <= purchased_up,
                f"sys_cap_reserve_up_{t_index}"
            )
            # Cap total down‐reserve by what system actually procured
            model.prob += (
                pulp.lpSum(model.pr_dn[(a, t_index)]
                        for (a, tt) in model.pr_dn if tt == t_index)
                <= purchased_dn,
                f"sys_cap_reserve_dn_{t_index}"
            )



    def build_objective(self, model):
        dt = model.dt
        for (a, t) in model.pr_up:
            r_up, r_dn, purchased_up, purchased_dn = self.reserve_prices.iloc[t]
            if r_up <= 0:
                model.prob += model.pr_up[(a, t)] == 0, f"no_pr_up_when_no_clear_{a}_{t}"
            if r_dn <= 0:
                model.prob += model.pr_dn[(a, t)] == 0, f"no_pr_dn_when_no_clear_{a}_{t}"
            # Only count capacity if clearing price > 0
            if r_up > 0:
                model.objective_terms.append(r_up * model.pr_up[(a, t)] * dt)
            # If r_up == 0, pr_up gets no revenue (and possibly won't be cleared).
            if r_dn > 0:
                model.objective_terms.append(r_dn * model.pr_dn[(a, t)] * dt)
