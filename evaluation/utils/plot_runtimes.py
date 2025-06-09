import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load CSV data
df = pd.read_csv('../results/plot_runtimes.csv')

# 2) Compute “total runtime” as the sum of all runtime_* columns
runtime_cols = [
    'runtime_simulation',
    'runtime_price_loading',
    'runtime_scheduling',
    'runtime_aggregation',
    'runtime_clustering'
]
df['runtime_total'] = df[runtime_cols].sum(axis=1)

# 3) Gather unique fleet sizes (NUM_EVS), sorted
num_evs = sorted(df['NUM_EVS'].unique())

# ----------------------------
# Subplot 1: runtime_total vs NUM_EVS for each ALIGNMENT (and DFO)
# ----------------------------
alignments = sorted(df[df['TYPE'] == 'FO']['ALIGNMENT'].unique())
data_align = {}

for a in alignments:
    means = [
        df[
            (df['TYPE'] == 'FO') &
            (df['ALIGNMENT'] == a) &
            (df['NUM_EVS'] == ne)
        ]['runtime_total'].mean()
        for ne in num_evs
    ]
    data_align[a] = means

# Add DFO as its own series
means_dfo = [
    df[
        (df['TYPE'] == 'DFO') &
        (df['NUM_EVS'] == ne)
    ]['runtime_total'].mean()
    for ne in num_evs
]
data_align['DFO'] = means_dfo

# ----------------------------
# Subplot 2: runtime_total vs NUM_EVS for each CLUSTER_METHOD
# ----------------------------
cluster_methods = sorted(df['CLUSTER_METHOD'].unique())
data_cluster = {}

for m in cluster_methods:
    means = [
        df[
            (df['CLUSTER_METHOD'] == m) &
            (df['NUM_EVS'] == ne)
        ]['runtime_total'].mean()
        for ne in num_evs
    ]
    data_cluster[m] = means

# ----------------------------
# Subplot 3: runtime_scheduling vs NUM_EVS for each FO MODE
# ----------------------------
fo_df = df[df['TYPE'] == 'FO']
modes = ['Joint', 'Sequential', 'Spot Only']
data_mode_sched = {}

for mode in modes:
    means = []
    for ne in num_evs:
        if mode == 'Joint':
            subset = fo_df[
                (fo_df['MODE'] == 'joint') &
                (fo_df['RUN_RESERVE'] == True) &
                (fo_df['RUN_ACTIVATION'] == True) &
                (fo_df['NUM_EVS'] == ne)
            ]
        elif mode == 'Sequential':
            subset = fo_df[
                (fo_df['MODE'] == 'sequential_reserve_first') &
                (fo_df['RUN_RESERVE'] == True) &
                (fo_df['RUN_ACTIVATION'] == True) &
                (fo_df['NUM_EVS'] == ne)
            ]
        else:  # Spot Only
            subset = fo_df[
                (fo_df['RUN_RESERVE'] == False) &
                (fo_df['RUN_ACTIVATION'] == False) &
                (fo_df['NUM_EVS'] == ne)
            ]
        means.append(subset['runtime_scheduling'].mean() if not subset.empty else np.nan)
    data_mode_sched[mode] = means

# ----------------------------
# Subplot 4: runtime_scheduling vs NUM_EVS for TYPE (FO vs DFO)
# ----------------------------
types = ['FO', 'DFO']
data_type_sched = {}

for t in types:
    means = [
        df[
            (df['TYPE'] == t) &
            (df['NUM_EVS'] == ne)
        ]['runtime_scheduling'].mean()
        for ne in num_evs
    ]
    data_type_sched[t] = means

# ----------------------------
# Plot everything in a 2×2 grid
# ----------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot 1: Alignment + DFO lines
ax1 = axs[0, 0]
for label, values in data_align.items():
    ax1.plot(num_evs, values, marker='o', label=label)
ax1.set_xlabel('Number of EVs')
ax1.set_ylabel('Mean Total Runtime (s)')
ax1.set_title('Total Runtime vs. Fleet Size by Alignment/DFO')
ax1.legend(title='Series')
# Grid: draw only on minor x‐ticks (every 10,000)
ax1.grid(which='major', axis='x', linestyle='--', alpha=0.3)
ax1.grid(which='major', axis='y', linestyle='--', alpha=0.3)


# Plot 2: Cluster Method lines
ax2 = axs[0, 1]
for label, values in data_cluster.items():
    ax2.plot(num_evs, values, marker='o', label=label)
ax2.set_xlabel('Number of EVs')
ax2.set_ylabel('Mean Total Runtime (s)')
ax2.set_title('Total Runtime vs. Fleet Size by Clustering Method')
ax2.legend(title='Cluster Method')

ax2.grid(which='major', axis='x', linestyle='--', alpha=0.3)
ax2.grid(which='major', axis='y', linestyle='--', alpha=0.3)

# We'll stack only these four components:
runtime_parts = [
    'runtime_clustering',
    'runtime_aggregation',
    'runtime_scheduling'
]

# 1) Compute the mean of each of these four components for every fleet size
component_means = {
    col: np.array([df[df['NUM_EVS'] == ne][col].mean() for ne in num_evs])
    for col in runtime_parts
}

# 2) Compute total of these four parts per fleet size (for percentage normalization)
total_non_sim = np.zeros(len(num_evs))
for col in runtime_parts:
    total_non_sim += component_means[col]

# 3) Compute simulation means (absolute) for annotation
sim_means = np.array([df[df['NUM_EVS'] == ne]['runtime_simulation'].mean() for ne in num_evs])

# 4) Convert the four parts into fractions so that each bar sums to 1.0
frac_means = {col: component_means[col] / total_non_sim for col in runtime_parts}

# -------------------------------------------------
# Plot 3: 100% Stacked Bar of Non-Simulation Components
# -------------------------------------------------
ax3 = axs[1, 0]

# Create categorical x-positions so bars are evenly spaced
x_pos = np.arange(len(num_evs))
bar_width = 0.6

bottom = np.zeros(len(num_evs))
colors = ['#55A868', '#C44E52', '#8172B2', '#CCB974']  # price loading, clustering, aggregation, scheduling

for idx, col in enumerate(runtime_parts):
    vals = frac_means[col]
    bars = ax3.bar(
        x_pos,
        vals,
        bar_width,
        bottom=bottom,
        color=colors[idx],
        label=col.replace('runtime_', '').replace('_', ' ').title(),
        edgecolor='white'
    )

    # Place percentage labels inside segments that are >5%
    for bar, frac_val in zip(bars, vals):
        if frac_val > 0.05:
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{frac_val * 100:.0f}%",
                ha='center',
                va='center',
                fontsize=8,
                color='white',
                weight='bold',
            )
    bottom += vals

# # Annotate each bar’s simulation time (absolute seconds) above the plot title
# for xi, sim_val in zip(x_pos, sim_means):
#     ax3.text(
#         xi,
#         1.07,                  # a bit above the 100% mark
#         f"Sim: {sim_val:.1f}s",# e.g. "Sim: 45.7s"
#         ha='center',
#         va='bottom',
#         fontsize=9,
#         color='black'
#     )

# # Annotate each bar’s non-simulation total absolute time just above 100%
# for xi, total_ns in zip(x_pos, total_non_sim):
#     ax3.text(
#         xi,
#         1.02,                  # just above the 100% mark
#         f"{total_ns:.1f}s",    # e.g. "36.7s"
#         ha='center',
#         va='bottom',
#         fontsize=9,
#         color='black'
#     )

# Configure axes, grid, and legend
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"{int(x):,}" for x in num_evs], rotation=0, fontsize=10)
ax3.set_xlabel('Number of EVs')
ax3.set_ylabel('Fraction of Non-Simulation Runtime')
ax3.set_ylim(0, 1.10)
ax3.set_title(
    'Pipeline Breakdown (Excluding Simulation)',
)
ax3.yaxis.grid(True, linestyle='--', alpha=0.4)
ax3.legend(
    title='Pipeline Part',
    bbox_to_anchor=(1.02, 1),
    loc='upper left',
    borderaxespad=0,
    fontsize=9,
    title_fontsize=10
)

# -------------------------------------------------
# Leave axs[1,1] empty (no Plot 4)
# -------------------------------------------------
axs[1, 1].axis('off')

# -------------------------------------------------
# Show the complete figure
# -------------------------------------------------
plt.show()