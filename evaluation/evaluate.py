import numpy as np
import matplotlib.pyplot as plt
from database.dataManager import fetch_all_prices
from optimization.DFOOptimizer import DFO_Optimization
from classes.DFO import DFO
from evaluation.metrics import evaluate_schedule, statistical_summary

TIME_RESOLUTION = 3600  # Ensure this is consistent with your other scripts

def get_cost_array_for_dfo(dfo, prices):
    num_timesteps = len(dfo.polygons)
    return prices[:num_timesteps]

def naive_baseline(dfo):
    total_energy_needed = dfo.min_total_energy
    num_timesteps = len(dfo.polygons)
    return [total_energy_needed / num_timesteps] * num_timesteps

def greedy_baseline(dfo, prices):
    total_energy_needed = dfo.min_total_energy
    num_timesteps = len(dfo.polygons)
    schedule = [0.0] * num_timesteps
    timestep_order = sorted(range(num_timesteps), key=lambda t: prices[t])
    
    energy_remaining = total_energy_needed
    for t in timestep_order:
        max_energy = max(p.y for p in dfo.polygons[t].points)
        allocation = min(max_energy, energy_remaining)
        schedule[t] = allocation
        energy_remaining -= allocation
        if energy_remaining <= 0:
            break
    return schedule

def full_evaluation():
    prices = fetch_all_prices()
    scenarios = generate_test_scenario()

    optimizer_results, naive_results, greedy_results = [], [], []

    for dfo in scenarios:
        # Optimizer
        cost_array = get_cost_array_for_dfo(dfo, prices)
        optimized_schedule = DFO_Optimization(dfo, cost_array)
        eval_optimized = evaluate_schedule(optimized_schedule, cost_array)
        optimizer_results.append(eval_optimized["total_cost"])

        # Naive Baseline
        naive_schedule = naive_baseline(dfo)
        eval_naive = evaluate_schedule(naive_schedule, cost_array)
        naive_results.append(eval_naive["total_cost"])

        # Greedy Baseline
        greedy_schedule = greedy_baseline(dfo, cost_array)
        eval_greedy = evaluate_schedule(greedy_schedule, cost_array)
        greedy_results.append(eval_greedy["total_cost"])

    print("Optimizer stats:", statistical_summary(optimizer_results))
    print("Naive baseline stats:", statistical_summary(naive_results))
    print("Greedy baseline stats:", statistical_summary(greedy_results))

if __name__ == "__main__":
    full_evaluation()
