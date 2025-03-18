import numpy as np

def evaluate_schedule(energy_schedule, prices):
    total_cost = sum(e * p for e, p in zip(energy_schedule, prices))
    peak_usage = max(energy_schedule)

    return {
        "total_cost": total_cost,
        "peak_usage": peak_usage,
    }

def statistical_summary(results):
    return {
        "mean": np.mean(results),
        "std_dev": np.std(results),
        "95%_confidence_interval": (
            np.mean(results) - 1.96 * np.std(results) / np.sqrt(len(results)),
            np.mean(results) + 1.96 * np.std(results) / np.sqrt(len(results))
        )
    }
