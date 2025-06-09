import pulp
import pandas as pd
from typing import List, Union, Dict
from flexoffer_logic import Flexoffer, DFO
from config import config
from optimization.markets.spot_market import SpotMarket
from optimization.markets.reserve_market import ReserveMarket
from optimization.markets.activation_market import ActivationMarket

import pulp
from config import config

class BaseOptimizer:
    def __init__(self, offers, spot_prices, reserve_prices, activation_prices, indicators, base_ts: float = None):
        self.offers = offers
        self.spot_prices = spot_prices
        self.reserve_prices = reserve_prices
        self.activation_prices = activation_prices
        self.indicators = indicators

        if base_ts is None:
            self.sim_start_ts = pd.to_datetime(config.SIMULATION_START_DATE).timestamp()
        else:
            self.sim_start_ts = base_ts

        self.T = len(spot_prices)
        self.dt = config.TIME_RESOLUTION / 3600.0
        self.offsets = [int((fo.get_est() - self.sim_start_ts) // config.TIME_RESOLUTION) for fo in offers]

        self.prob = pulp.LpProblem("MarketOptimization", pulp.LpMaximize)
        self.p = {}
        self.pr_up = {}
        self.pr_dn = {}
        self.pb_up = {}
        self.pb_dn = {}
        self.s_up = {}
        self.s_dn = {}
        self.objective_terms = []

    def build_modules(self):
        modules = []
        modules.append(SpotMarket(self.spot_prices))
        modules.append(ReserveMarket(self.reserve_prices))
        modules.append(ActivationMarket(self.activation_prices, self.indicators))
        return modules

    def run_joint(self):
        modules = self.build_modules()
        for m in modules: m.create_variables(self)
        for m in modules: m.add_constraints(self)
        for m in modules: m.build_objective(self)
        return self.solve()
    
    def run_sequential_reserve_first(self):
        # ---------- PHASE 1: Spot feasibility + Reserve only ----------
        # 1. Create and constrain baseline p variables
        spot_mod = SpotMarket(self.spot_prices)
        spot_mod.create_variables(self)
        spot_mod.add_constraints(self)    # enforce per-slice AND total-energy constraints

        # 2. Create & constrain reserve variables on that baseline
        res_mod = ReserveMarket(self.reserve_prices)
        res_mod.create_variables(self)
        res_mod.add_constraints(self)
        res_mod.build_objective(self)     # maximize reserve revenue
        self.solve()

        # 3. Capture the fixed values of p, pr_up, pr_dn
        fixed_p     = {k: v.value() for k, v in self.p.items()}
        fixed_pr_up = {k: v.value() for k, v in self.pr_up.items()}
        fixed_pr_dn = {k: v.value() for k, v in self.pr_dn.items()}

        # ---------- PHASE 2: Re‐init model, re‐fix vars, then add all objectives ----------
        # Re‐initialize the optimizer to clear the old LP
        self.__init__(self.offers,
                      self.spot_prices,
                      self.reserve_prices,
                      self.activation_prices,
                      self.indicators)

        # 4. Re-create variables and immediately fix them to Phase 1 values
        #    Baseline p
        spot_mod = SpotMarket(self.spot_prices)
        spot_mod.create_variables(self)
        for (a, t), val in fixed_p.items():
            var = self.p[(a, t)]
            var.lowBound = val
            var.upBound  = val

        #    Reserve bids pr_up/pr_dn
        res_mod = ReserveMarket(self.reserve_prices)
        res_mod.create_variables(self)
        for (a, t), val in fixed_pr_up.items():
            var = self.pr_up[(a, t)]
            var.lowBound = val
            var.upBound  = val
        for (a, t), val in fixed_pr_dn.items():
            var = self.pr_dn[(a, t)]
            var.lowBound = val
            var.upBound  = val

        # 5. Re-add spot & reserve objective terms on those fixed vars
        spot_mod.build_objective(self)
        res_mod.build_objective(self)

        # 6. Create & constrain activation variables, then add its objective
        if config.RUN_ACTIVATION:
            act_mod = ActivationMarket(self.activation_prices, self.indicators)
            act_mod.create_variables(self)
            act_mod.add_constraints(self)
            act_mod.build_objective(self)

        # 7. Solve the fully‐assembled LP
        return self.solve()


    def run_theoretical_optimum(self):
        original_mode = config.MODE
        original_spot = config.RUN_SPOT
        original_res = config.RUN_RESERVE
        original_act = config.RUN_ACTIVATION

        config.MODE = "joint"
        config.RUN_SPOT = True
        config.RUN_RESERVE = True
        config.RUN_ACTIVATION = True

        result = self.run_joint()

        config.MODE = original_mode
        config.RUN_SPOT = original_spot
        config.RUN_RESERVE = original_res
        config.RUN_ACTIVATION = original_act

        return result

    def solve(self):
        self.prob += pulp.lpSum(self.objective_terms)
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return self.extract_solution()

    def extract_solution(self):
        A = len(self.offers)
        sol = {k: {a: {} for a in range(A)} for k in ["p", "pr_up", "pr_dn", "pb_up", "pb_dn", "s_up", "s_dn"]}
        for (a, t), var in self.p.items(): sol["p"][a][t] = pulp.value(var)
        for (a, t), var in self.pr_up.items(): sol["pr_up"][a][t] = pulp.value(var)
        for (a, t), var in self.pr_dn.items(): sol["pr_dn"][a][t] = pulp.value(var)
        for (a, t), var in self.pb_up.items(): sol["pb_up"][a][t] = pulp.value(var)
        for (a, t), var in self.pb_dn.items(): sol["pb_dn"][a][t] = pulp.value(var)
        for (a, t), var in self.s_up.items(): sol["s_up"][a][t] = pulp.value(var)
        for (a, t), var in self.s_dn.items(): sol["s_dn"][a][t] = pulp.value(var)
        return sol

    def run(self):
        if config.MODE == "joint":
            return self.run_joint()
        elif config.MODE == "sequential_reserve_first":
            return self.run_sequential_reserve_first()
        else:
            raise NotImplementedError(f"Unsupported mode: {config.MODE}")
