import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "../results/pct_%ofoptimal.csv"

df = pd.read_csv(RESULTS_DIR)
import pandas as pd
import matplotlib.pyplot as plt

# 2) Compute the mean % of theoretical optimum for each unique scenario:
group_keys = [
    'TYPE', 'MODE',
    'RUN_RESERVE', 'RUN_ACTIVATION',
    'CLUSTER_METHOD',
    'NUM_CLUSTERS'
]
mean_df = (
    df
    .groupby(group_keys)['pct_of_optimal']
    .mean()
    .reset_index()
)

# 3) Identify the distinct values of TYPE, MODE, CLUSTER_METHOD, and NUM_CLUSTERS
types            = ['FO', 'DFO']
modes            = ['joint', 'sequential_reserve_first']
cluster_methods  = mean_df['CLUSTER_METHOD'].unique()
num_clusters     = sorted(mean_df['NUM_CLUSTERS'].unique())

# 4) Create a 2×2 grid of subplots (rows=TYPE, columns=MODE), sharing the y-axis:
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)

for i, t in enumerate(types):
    for j, m in enumerate(modes):
        ax = axes[i, j]
        is_joint = (m == 'joint')

        for cm in cluster_methods:
            if is_joint:
                # In “joint” mode, plot two curves per clustering method:
                #  - RUN_RESERVE=False (and RUN_ACTIVATION=False)
                #  - RUN_RESERVE=True  (and RUN_ACTIVATION=True)
                for run_res in [False, True]:
                    subset = mean_df[
                        (mean_df['TYPE']            == t) &
                        (mean_df['MODE']            == m) &
                        (mean_df['RUN_RESERVE']     == run_res) &
                        (mean_df['RUN_ACTIVATION']  == run_res) &
                        (mean_df['CLUSTER_METHOD']  == cm)
                    ]
                    if subset.empty:
                        continue
                    ax.plot(
                        subset['NUM_CLUSTERS'],
                        subset['pct_of_optimal'],
                        marker='o',
                        label=f"{cm}, Reserve={run_res}"
                    )
            else:
                # In “sequential_reserve_first” mode, reserve+activation is always True.
                subset = mean_df[
                    (mean_df['TYPE']           == t) &
                    (mean_df['MODE']           == m) &
                    (mean_df['CLUSTER_METHOD'] == cm)
                ]
                if subset.empty:
                    continue
                ax.plot(
                    subset['NUM_CLUSTERS'],
                    subset['pct_of_optimal'],
                    marker='o',
                    label=cm
                )

        # Use logarithmic scale on the x-axis (NUM_CLUSTERS spans [2, 5, 10, 20])
        ax.set_xscale('log')
        ax.set_xticks(num_clusters)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        ax.set_title(f"Type = {t}, Mode = {m}")
        ax.set_xlabel("Number of Clusters")
        if j == 0:
            ax.set_ylabel("% of Theoretical Optimum")
        ax.legend(fontsize='small')

fig.tight_layout()
plt.show()

