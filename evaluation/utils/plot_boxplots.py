import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load your DataFrame
df = pd.read_csv('../results/plot_boxplots.csv')

# Define the metric
metric_col = 'pct_saved_vs_greedy_baseline'

# --- Subplot 1: FO by Alignment + Single DFO ---

# Unique alignment values, sorted (for FO)
alignments = sorted(df['ALIGNMENT'].unique())

# Separate FO and DFO entries
fo_df = df[df['TYPE'] == 'FO']
dfo_df = df[df['TYPE'] == 'DFO']

# Prepare FO data lists: for each alignment, collect metric values
fo_data = [
    fo_df[fo_df['ALIGNMENT'] == a][metric_col].dropna().values
    for a in alignments
]

# Prepare a single DFO data list (all DFO values regardless of alignment)
#dfo_data_single = [dfo_df[metric_col].dropna().values]

# --- Subplot 2: FO Savings by Mode ---

# Filter only FO type for mode‐based plot
fo_mode_df = df[df['TYPE'] == 'FO']

# Define groups for FO modes:
# 1) Joint: MODE == 'joint' & RUN_RESERVE=True & RUN_ACTIVATION=True
joint_df = fo_mode_df[
    (fo_mode_df['MODE'] == 'joint') &
    (fo_mode_df['RUN_RESERVE'] == True) &
    (fo_mode_df['RUN_ACTIVATION'] == True)
]

# 2) Sequential: MODE == 'sequential_reserve_first' & RUN_RESERVE=True & RUN_ACTIVATION=True
seq_df = fo_mode_df[
    (fo_mode_df['MODE'] == 'sequential_reserve_first') &
    (fo_mode_df['RUN_RESERVE'] == True) &
    (fo_mode_df['RUN_ACTIVATION'] == True)
]

# 3) Spot Only: RUN_RESERVE=False & RUN_ACTIVATION=False
spot_only_df = fo_mode_df[
    (fo_mode_df['RUN_RESERVE'] == False) &
    (fo_mode_df['RUN_ACTIVATION'] == False)
]

# Extract the pct_total_saved values for each mode
joint_data       = joint_df[metric_col].dropna().values
seq_data         = seq_df[metric_col].dropna().values
spot_only_data   = spot_only_df[metric_col].dropna().values

# Plot both subplots side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
plt.subplots_adjust(wspace=0.4)

# --- Plot 1: FO by Alignment + DFO ---
ax1 = axes[0]
width = 0.6  # width for boxplots

# FO boxplots (offset left)
x_positions_fo = np.arange(len(alignments))
ax1.boxplot(
    fo_data,
    positions=x_positions_fo,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor='lightblue', edgecolor='blue', linewidth=1.5),
    medianprops=dict(color='blue', linewidth=2),
    whiskerprops=dict(color='blue', linewidth=1.5),
    capprops=dict(color='blue', linewidth=1.5),
    flierprops=dict(marker='o', markeredgecolor='blue', markerfacecolor='white', markersize=5)
)

# Single DFO boxplot (offset right)
# x_position_dfo = len(alignments)
# ax1.boxplot(
#     dfo_data_single,
#     positions=[x_position_dfo],
#     widths=width,
#     patch_artist=True,
#     boxprops=dict(facecolor='lightgreen', edgecolor='green', linewidth=1.5),
#     medianprops=dict(color='green', linewidth=2),
#     whiskerprops=dict(color='green', linewidth=1.5),
#     capprops=dict(color='green', linewidth=1.5),
#     flierprops=dict(marker='o', markeredgecolor='green', markerfacecolor='white', markersize=5)
# )

# Set x‐ticks and labels for subplot 1
xticks1 = list(x_positions_fo)# + [x_position_dfo]
xtick_labels1 = alignments #+ ['DFO']
ax1.set_xticks(xticks1)
ax1.set_xticklabels(xtick_labels1, rotation=45, ha='right')
ax1.set_xlabel('Alignment (FO)')
ax1.set_ylabel('Savings Compared to Baseline (%)')
ax1.set_title('FO Savings by Alignment')

# Custom legend for subplot 1
legend_handles1 = [
    plt.Line2D([0], [0], color='lightblue', lw=10, label='FO by Alignment'),
  #  plt.Line2D([0], [0], color='lightgreen', lw=10, label='DFO Aggregation')
]
ax1.legend(handles=legend_handles1, loc='upper left')

# --- Plot 2: FO Savings by Mode ---
ax2 = axes[1]
positions2 = [1, 2, 3]
data_to_plot2 = [joint_data, seq_data, spot_only_data]
labels2 = ['Joint', 'Sequential', 'Spot Only']
colors2 = ['skyblue', 'lightcoral', 'lightgreen']

for pos, data, color in zip(positions2, data_to_plot2, colors2):
    ax2.boxplot(
        data,
        positions=[pos],
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor=color, edgecolor='black', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        flierprops=dict(marker='o', markeredgecolor='black', markerfacecolor='white', markersize=5)
    )

ax2.set_xticks(positions2)
ax2.set_xticklabels(labels2)
ax2.set_xlabel('Operational Mode')
ax2.set_ylabel('Savings Compared to Baseline (%)')
ax2.set_title('Distribution of FO Savings by Mode')

plt.tight_layout()
plt.show()