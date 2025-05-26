from typing import List, Dict, Optional
import pulp
from flexoffer_logic import Flexoffer, TimeSlice
import pandas as pd
import logging
from config import config

logger = logging.getLogger(__name__)

class FO_Opt:
    """
    A simplified FlexOffer optimizer for spot market scheduling.
    
    Attributes:
        offers (List[Flexoffer]): List of FlexOffer objects to schedule.
        spot_prices (pd.Series): Spot market prices indexed by DateTime timestamps.
    """
    
    def __init__(self, 
                 offers: List[Flexoffer],
                 spot_prices: pd.Series):
        # FlexOffers and corresponding spot prices (spot prices start at simulation start date from config)
        self.offers = offers
        self.spot_prices = spot_prices

        # Simulation resolution and start time
        self.res = config.TIME_RESOLUTION
        self.sim_start_ts = int(pd.to_datetime(config.SIMULATION_START_DATE).timestamp())
        
        # Time horizon (number of price slots)
        self.T = len(spot_prices)
        self.dt = self.res / 3600  # time step in hours

        # Dictionary to hold decision variables for power in spot market
        self.power = {}

        # Linear optimization problem instance
        self.prob = pulp.LpProblem("FO_Scheduling", pulp.LpMaximize)

    def create_variables(self):
        """
        Initializes power variables for each FlexOffer at each relevant time index.
        Variables are constrained by the offer's min/max power per time slice.
        """
        for a, fo in enumerate(self.offers):
            prof = fo.get_profile()
            for j, ts in enumerate(prof):
                # Compute offset from simulation start to FlexOffer start
                # important for proper indexing when we create decision variables
                offset = int((fo.get_est() - self.sim_start_ts) / self.res)
                t = offset + j
                # Create a decision power variable for time index t
                self.power[(a, t)] = pulp.LpVariable(f"p_{a}_{t}", lowBound=0, upBound=ts.max_power)

    def add_constraints(self):
        """
        Adds energy constraints per FlexOffer:
        - Total allocated energy must lie within [min_alloc, max_alloc].
        """
        for a, fo in enumerate(self.offers):
            prof = fo.get_profile()
            terms = []
            for j, ts in enumerate(prof):
                t = int((fo.get_est() - self.sim_start_ts) / self.res) + j
                # Each time slice contributes power × time to the total energy
                terms.append(self.power[(a,t)] * self.dt)

            # Add lower and upper bound constraints
            self.prob += pulp.lpSum(terms) >= fo.get_min_overall_alloc()
            self.prob += pulp.lpSum(terms) <= fo.get_max_overall_alloc()

    def build_objective(self):
        """
        Builds the profit-maximization objective based on spot prices.
        Lower spot prices yield higher cost (hence negative term).
        """
        obj = []
        for (a, t), var in self.power.items():
            spot = self.spot_prices.iloc[t]
            obj.append(-spot * var * self.dt)
        self.prob += pulp.lpSum(obj)

    def solve(self):
        """
        Solves the MILP problem using CBC solver (default via PuLP).
        """
        self.prob.solve(pulp.PULP_CBC_CMD(msg=False))

    def extract_solution(self):
        """
        Extracts the optimal schedule and updates each FlexOffer with:
        - scheduled allocation per timeslice
        """
        for a, fo in enumerate(self.offers):
            if fo.get_scheduled_allocation() is not None:
                allocation = [0.0] * fo.get_duration()
                prof = fo.get_profile()
                for j, ts in enumerate(prof):
                    t = int((fo.get_est() - self.sim_start_ts) / self.res) + j
                    if (a, t) in self.power:
                        allocation[j] = pulp.value(self.power[(a, t)])
                fo.set_scheduled_allocation(allocation)

    def run(self):
        """
        Runs the full optimization pipeline: define variables, add constraints,
        build the objective, solve the LP, and write solution into offers.
        """
        self.create_variables()
        self.add_constraints()
        self.build_objective()
        self.solve()
        self.extract_solution()
