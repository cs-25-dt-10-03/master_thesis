import pandas as pd
import matplotlib.pyplot as plt
from plot_style import ALIGNMENT_MARKERS, ALIGNMENT_STYLES  # Make sure these are defined

# Load only spot-enabled configurations
df = pd.read_csv("../results/run_summary.csv")
df = df[df['RUN_SPOT']]

df["runtime_scheduling"] = pd.to_numeric(df["runtime_scheduling"], errors="coerce")
df["runtime_aggregation"] = pd.to_numeric(df["runtime_aggregation"], errors="coerce")

def plot_runtimes_vs_num_evs(data):
    sched = data.groupby(["NUM_EVS", "ALIGNMENT"])["runtime_scheduling"].mean().reset_index()
    agg   = data.groupby(["NUM_EVS", "ALIGNMENT"])["runtime_aggregation"].mean().reset_index()

    # Plot scheduling runtime
    plt.figure(figsize=(10, 6))
    for alignment in sched['ALIGNMENT'].unique():
        sub = sched[sched['ALIGNMENT'] == alignment]
        plt.plot(
            sub['NUM_EVS'],
            sub['runtime_scheduling'],
            label=f"Scheduling | {alignment}",
            linestyle=ALIGNMENT_STYLES[alignment],
            marker=ALIGNMENT_MARKERS[alignment]
        )
    plt.title("Scheduling Runtime vs Number of FlexOffers")
    plt.xlabel("Number of FlexOffers (NUM_EVS)")
    plt.ylabel("Runtime Scheduling (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot aggregation runtime
    plt.figure(figsize=(10, 6))
    for alignment in agg['ALIGNMENT'].unique():
        sub = agg[agg['ALIGNMENT'] == alignment]
        plt.plot(
            sub['NUM_EVS'],
            sub['runtime_aggregation'],
            label=f"Aggregation | {alignment}",
            linestyle=ALIGNMENT_STYLES[alignment],
            marker=ALIGNMENT_MARKERS[alignment]
        )
    plt.title("Aggregation Runtime vs Number of FlexOffers")
    plt.xlabel("Number of FlexOffers (NUM_EVS)")
    plt.ylabel("Runtime Aggregation (s)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_runtimes_vs_num_evs(df)
