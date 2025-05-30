import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "../results/synth_summary.csv"

df = pd.read_csv(RESULTS_DIR)

df = df[df['RUN_SPOT'].fillna(False)]

# Create a human‑readable market scenario label
def market_label(row):
    if row['RUN_RESERVE'] and row['RUN_ACTIVATION']:
        return 'Spot + Reserve + Activation'
    elif row['RUN_RESERVE']:
        return 'Spot + Reserve'
    else:
        return 'Spot only'
df['market'] = df.apply(market_label, axis=1)

# Prepare scaling data
sched_scaling = df.groupby(['NUM_EVS','TIME_RES','MODE'])['runtime_scheduling']\
                  .mean().reset_index()
agg_scaling   = df.groupby(['NUM_EVS','TIME_RES'])['runtime_aggregation']\
                  .mean().reset_index()

# Time resolution mapping
res_labels = {900: '15 min', 3600: '60 min'}

# Set up 2×2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top‑left: Scheduling scaling
ax = axes[0, 0]
for (res, mode), grp in sched_scaling.groupby(['TIME_RES','MODE']):
    ax.plot(grp['NUM_EVS'], grp['runtime_scheduling'],
            marker='o' if res==900 else 's',
            linestyle='-' if mode=='joint' else '--',
            label=f"{mode.title()} @ {res_labels[res]}")
ax.set_xlabel('Number of FlexOffers')
ax.set_ylabel('Scheduling Runtime (s)')
ax.set_title('Scheduling Runtime vs FlexOffers')
ax.legend()

# Top‑right: Aggregation scaling
ax = axes[0, 1]
for res, grp in agg_scaling.groupby('TIME_RES'):
    ax.plot(grp['NUM_EVS'], grp['runtime_aggregation'],
            marker='o' if res==900 else 's',
            linestyle='-',
            label=f"Aggregation @ {res_labels[res]}")
ax.set_xlabel('Number of FlexOffers')
ax.set_ylabel('Aggregation Runtime (s)')
ax.set_title('Aggregation Runtime vs FlexOffers')
ax.legend()

# Bottom‑left: Scheduling by market @ 15 min
ax = axes[1, 0]
data15 = df[df['TIME_RES']==900]\
    .groupby(['market','MODE'])['runtime_scheduling']\
    .mean().unstack('MODE')
data15.plot(kind='bar', ax=ax)
ax.set_xlabel('')
ax.set_ylabel('Scheduling Runtime (s)')
ax.set_title('Scheduling Runtime by Market (15 min)')
ax.legend(title='Mode')
ax.set_xticklabels(data15.index, rotation=45, ha='right')

# Bottom‑right: Scheduling by market @ 60 min
ax = axes[1, 1]
data60 = df[df['TIME_RES']==3600]\
    .groupby(['market','MODE'])['runtime_scheduling']\
    .mean().unstack('MODE')
data60.plot(kind='bar', ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Scheduling Runtime by Market (60 min)')
ax.legend(title='Mode')
ax.set_xticklabels(data60.index, rotation=45, ha='right')

plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Fallback styling if plot_style.py is not accessible
MODE_STYLES = {"joint": "-", "sequential": "--", "sequential_reserve_first": ":"}
MODE_MARKERS = {"joint": "o", "sequential": "s", "sequential_reserve_first": "v"}

df = pd.read_csv(RESULTS_DIR)

def market_label(row):
    if row['RUN_RESERVE'] and row['RUN_ACTIVATION']:
        return 'Spot + Reserve + Activation'
    elif row['RUN_RESERVE']:
        return 'Spot + Reserve'
    else:
        return 'Spot only'

df['Market']   = df.apply(market_label, axis=1)
df['Scenario'] = df['Market'] + ' | ' + df['MODE'].str.title()


res_labels  = {900: '15 min', 3600: '60 min'}
metrics = [
    ('pct_total_saved', 'Total Savings (%)'),
    ('pct_of_optimal', '% of Theoretical Maximum')
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey='row')

for i, (metric, ylabel) in enumerate(metrics):
    for j, res in enumerate([900, 3600]):
        ax = axes[i][j]
        sub = df[df['TIME_RES'] == res]

        # Spot only grouped properly
        spot = (
            sub[(~sub['RUN_RESERVE']) & (sub['MODE'] == "joint")]
            .groupby('NUM_CLUSTERS')[metric]
            .mean()
            .reset_index()
        )
        if not spot.empty:
            ax.plot(
                spot['NUM_CLUSTERS'],
                spot[metric],
                label='Spot only',
                linestyle='-',
                marker='o',
                color='red'
            )

        others = (
            sub[(sub['RUN_RESERVE']) & (sub['RUN_ACTIVATION'])]
            .groupby(['NUM_CLUSTERS', 'MODE'])[metric]
            .mean()
            .reset_index()
        )
        for mode in others['MODE'].unique():
            grp = others[others['MODE'] == mode].sort_values('NUM_CLUSTERS')
            ax.plot(
                grp['NUM_CLUSTERS'],
                grp[metric],
                label=f"Spot + Reserve + Activation | {mode.title()}",
                linestyle=MODE_STYLES[mode],
                marker=MODE_MARKERS[mode]
            )

        ax.set_title(res_labels[res])
        ax.set_xlabel('Number of Clusters')
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.grid(True)

axes[0][1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

