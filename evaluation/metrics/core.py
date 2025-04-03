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

def summarize_metrics(all_results):
    summary = {}
    for method in ["optimizer", "greedy", "naive"]:
        method_costs = [r[method]["total_cost"] for r in all_results]
        method_energies = [r[method]["total_energy"] for r in all_results]

        summary[method] = {
            "mean_cost": float(np.mean(method_costs)),
            "std_cost": float(np.std(method_costs)),
            "95%_ci_cost": (
                float(np.mean(method_costs) - 1.96 * np.std(method_costs) / np.sqrt(len(method_costs))),
                float(np.mean(method_costs) + 1.96 * np.std(method_costs) / np.sqrt(len(method_costs)))
            ),
            "mean_energy": float(np.mean(method_energies)),
        }
    return summary
