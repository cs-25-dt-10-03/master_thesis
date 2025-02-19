class BaseOptimizer:
    def __init__(self, flex_offers, spot_price_data, config):
        self.flex_offers = flex_offers
        self.spot_price_data = spot_price_data
        self.config = config
        self.model = None
        

    def build_model(self):
        raise NotImplementedError("build_model must be implemented in the subclass")

    def solve(self):
        raise NotImplementedError("solve must be implemented in the subclass")

    def update_schedule(self):
        raise NotImplementedError("update_schedule must be implemented in the subclass")