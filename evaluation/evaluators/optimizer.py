from optimization.DFOOptimizer import DFO_Optimization
from flexoffer_logic import DFO, Flexoffer

def run_optimizer(obj, prices):
    if isinstance(obj, DFO):
        return DFO_Optimization(obj, prices)
    elif isinstance(obj, Flexoffer):
        duration = obj.get_duration()
        energy = obj.get_min_overall_alloc()
        return [energy / duration] * duration
    else:
        raise TypeError("Unsupported type for optimizer")
