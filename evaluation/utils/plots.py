import matplotlib.pyplot as plt

def plot_results(all_results):
    for i, result in enumerate(all_results):
        plt.figure(figsize=(10, 5))
        for method in ["optimizer", "greedy", "naive"]:
            schedule = result[method]["schedule"]
            if schedule:
                plt.plot(schedule, label=method)

        plt.title(f"Energy Schedules for Scenario {i}")
        plt.xlabel("Time step")
        plt.ylabel("Energy allocation (kWh)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
