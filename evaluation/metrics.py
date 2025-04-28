import numpy as np

def evaluate_schedule(schedule, prices):
    total_cost = sum(e * p for e, p in zip(schedule, prices))
    peak_usage = max(schedule) if schedule else 0
    total_energy = sum(schedule)

    return {
        "schedule": schedule,
        "total_cost": total_cost,
        "total_energy": total_energy,
        "peak_usage": peak_usage,
    }