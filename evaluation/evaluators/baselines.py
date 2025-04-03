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
    else:
        raise TypeError("Unsupported type for naive baseline")

def run_greedy(obj, prices):
    if isinstance(obj, DFO):
        total_energy = obj.min_total_energy
        schedule = [0.0] * len(prices)
        order = sorted(range(len(prices)), key=lambda i: prices[i])
        remaining = total_energy

        for i in order:
            max_val = 1.0  # Placeholder: could be max from polygon
            allocation = min(max_val, remaining)
            schedule[i] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
        return schedule

    elif isinstance(obj, Flexoffer):
        duration = obj.get_duration()
        energy = obj.get_min_overall_alloc()
        schedule = [0.0] * duration
        order = sorted(range(duration), key=lambda i: prices[i])
        remaining = energy

        for i in order:
            max_val = obj.get_profile()[i].max_power
            allocation = min(max_val, remaining)
            schedule[i] = allocation
            remaining -= allocation
            if remaining <= 0:
                break
        return schedule

    else:
        raise TypeError("Unsupported type for greedy baseline")
