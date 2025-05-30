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
    def __init__(self, offers, spot_prices, reserve_prices, activation_prices, indicators):
        self.offers = offers
        self.spot_prices = spot_prices
        self.reserve_prices = reserve_prices
        self.activation_prices = activation_prices
        self.indicators = indicators

        self.T = len(spot_prices)
        self.dt = config.TIME_RESOLUTION / 3600.0
       
        self.sim_start_ts = int(pd.to_datetime(config.SIMULATION_START_DATE).timestamp())
        self.offsets = [int((fo.get_est() - self.sim_start_ts) / config.TIME_RESOLUTION) for fo in offers]

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
        
        # Phase 1: Create spot variables (for reserve feasibility)
        SpotMarket(self.spot_prices).create_variables(self)

        # Phase 1: Solve reserve allocation
        ReserveMarket(self.reserve_prices).create_variables(self)
        ReserveMarket(self.reserve_prices).add_constraints(self)
        ReserveMarket(self.reserve_prices).build_objective(self)
        self.solve()

        fixed_p = self.p.copy()
        fixed_pr_up = self.pr_up.copy()
        fixed_pr_dn = self.pr_dn.copy()

        # Phase 2: Reset and re-initialize for spot + activation
        self.__init__(self.offers, self.spot_prices, self.reserve_prices, self.activation_prices, self.indicators)

        # Recreate fixed p and reserve vars
        for (a, t), val in fixed_p.items():
            var = pulp.LpVariable(f"p_{a}_{t}", lowBound=pulp.value(val), upBound=pulp.value(val))
            self.p[(a, t)] = var
        for (a, t), val in fixed_pr_up.items():
            var = pulp.LpVariable(f"pr_up_{a}_{t}", lowBound=pulp.value(val), upBound=pulp.value(val))
            self.pr_up[(a, t)] = var
        for (a, t), val in fixed_pr_dn.items():
            var = pulp.LpVariable(f"pr_dn_{a}_{t}", lowBound=pulp.value(val), upBound=pulp.value(val))
            self.pr_dn[(a, t)] = var

        # Phase 2: Solve spot + activation
        modules = []
        if config.RUN_SPOT:
            modules.append(SpotMarket(self.spot_prices))
        if config.RUN_ACTIVATION:
            modules.append(ActivationMarket(self.activation_prices, self.indicators))

        for m in modules:
            m.create_variables(self)
            m.add_constraints(self)
            m.build_objective(self)

        return self.solve()


    def run_theoretical_optimum(self):
        original_mode = config.MODE
        config.MODE = "joint"
        original_spot = config.RUN_SPOT
        original_res = config.RUN_RESERVE
        original_act = config.RUN_ACTIVATION

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
