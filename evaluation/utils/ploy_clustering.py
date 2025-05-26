
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import MARKET_COLORS

df = pd.read_csv("../results/synthetic_summary.csv")

# Grouped means for runtime
runtime_by_method = df.groupby("CLUSTER_METHOD")["runtime_aggregation"].mean().reset_index()
runtime_by_dynamic = df.groupby("DYNAMIC_CLUSTERING")["runtime_aggregation"].mean().reset_index()
runtime_by_parallel = df.groupby("PARALLEL_CLUSTER_AGGREGATION")["runtime_aggregation"].mean().reset_index()
interaction = df.groupby(["CLUSTER_METHOD", "DYNAMIC_CLUSTERING"])["runtime_aggregation"].mean().reset_index()

# Setup: 2x2 bar+scatter figure for runtime
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Runtime by CLUSTER_METHOD (bar)
axes[0, 0].bar(runtime_by_method["CLUSTER_METHOD"], runtime_by_method["runtime_aggregation"], color="#9ecae1")
axes[0, 0].set_title("Aggregation Runtime by CLUSTER_METHOD")
axes[0, 0].set_ylabel("Runtime (s)")

# Plot 2: Runtime by DYNAMIC_CLUSTERING (bar)
axes[0, 1].bar(["Off", "On"], runtime_by_dynamic["runtime_aggregation"], color=["#c6dbef", "#2171b5"])
axes[0, 1].set_title("Impact of DYNAMIC_CLUSTERING")
axes[0, 1].set_ylabel("Runtime (s)")

# Plot 3: Runtime by PARALLEL_CLUSTER_AGGREGATION (bar)
axes[1, 0].bar(["Off", "On"], runtime_by_parallel["runtime_aggregation"], color=["#fdae6b", "#e6550d"])
axes[1, 0].set_title("Impact of PARALLEL_CLUSTER_AGGREGATION")
axes[1, 0].set_ylabel("Runtime (s)")

# Plot 4: Interaction — CLUSTER_METHOD vs DYNAMIC_CLUSTERING (scatter)
for dynamic in [False, True]:
    subset = interaction[interaction["DYNAMIC_CLUSTERING"] == dynamic]
    label = "Dynamic On" if dynamic else "Dynamic Off"
    color = "#1b9e77" if dynamic else "#d95f02"
    axes[1, 1].scatter(subset["CLUSTER_METHOD"], subset["runtime_aggregation"], label=label, s=100, color=color)

axes[1, 1].set_title("Interaction: Method vs Dynamic Clustering")
axes[1, 1].set_ylabel("Mean Aggregation Runtime")
axes[1, 1].legend(title="Dynamic")

plt.tight_layout()
plt.show()

# --------- Economic Impact (bar only) ---------

savings_by_method = df.groupby("CLUSTER_METHOD")["pct_total_saved"].mean().reset_index()
savings_by_dynamic = df.groupby("DYNAMIC_CLUSTERING")["pct_total_saved"].mean().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar: Savings by CLUSTER_METHOD
axes[0].bar(savings_by_method["CLUSTER_METHOD"], savings_by_method["pct_total_saved"], color="#a1d99b")
axes[0].set_title("Savings by CLUSTER_METHOD")
axes[0].set_ylabel("Total Saved (%)")

# Bar: Savings by DYNAMIC_CLUSTERING
axes[1].bar(["Off", "On"], savings_by_dynamic["pct_total_saved"], color=["#c7e9c0", "#238b45"])
axes[1].set_title("Impact of DYNAMIC_CLUSTERING on Savings")
axes[1].set_ylabel("Total Saved (%)")

plt.tight_layout()
plt.show()