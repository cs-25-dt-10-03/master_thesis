from flexoffer_logic import DFO, Flexoffer

def run_naive(obj, prices):
    if isinstance(obj, DFO):
        total_energy = obj.min_total_energy
        num_steps = len(prices)
        return [total_energy / num_steps] * num_steps
    elif isinstance(obj, Flexoffer):
        duration = obj.get_duration()
        energy = obj.get_min_overall_alloc()
        return [energy / duration] * duration
