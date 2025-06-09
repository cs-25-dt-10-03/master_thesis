import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv('../results/plot_boxplots.csv')

# Columns for market contributions
contribution_columns = ['pct_saved_spot', 'pct_saved_spot', 'pct_gain_res', 'pct_gain_act'][1:]  # corrected for clarity
contribution_columns = ['pct_saved_spot', 'pct_gain_res', 'pct_gain_act']

# Prepare figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# 1. Overall contributions
data_overall = [df[col].dropna() for col in contribution_columns]
axes[0, 0].boxplot(data_overall, labels=['Spot', 'Reserve', 'Activation'])
axes[0, 0].set_title('Overall Market Contributions')
axes[0, 0].set_ylabel('Percentage')

# 2. FO only
df_fo = df[df['TYPE'] == 'FO']
data_fo = [df_fo[col].dropna() for col in contribution_columns]
axes[0, 1].boxplot(data_fo, labels=['Spot', 'Reserve', 'Activation'])
axes[0, 1].set_title('FO Market Contributions')

# 3. DFO only
df_dfo = df[df['TYPE'] == 'DFO']
data_dfo = [df_dfo[col].dropna() for col in contribution_columns]
axes[1, 0].boxplot(data_dfo, labels=['Spot', 'Reserve', 'Activation'])
axes[1, 0].set_title('DFO Market Contributions')
axes[1, 0].set_ylabel('Percentage')

# 4. By MODE (joint vs sequential_reserve_first)
modes = ['joint', 'sequential_reserve_first']
data_by_mode = []
labels_by_mode = []
for mode in modes:
    df_mode = df[df['MODE'] == mode]
    for col in contribution_columns:
        data_by_mode.append(df_mode[col].dropna())
        if mode == "joint":
            if col == "pct_saved_spot":
                labels_by_mode.append("Joint Spot Saved")
            elif col == "pct_gain_res":
                labels_by_mode.append("Joint Reserve Saved")
            else:  # pct_gain_act
                labels_by_mode.append("Joint Activation Gain")
        else:  # sequential_reserve_first
            if col == "pct_saved_spot":
                labels_by_mode.append("Sequential Spot Saved")
            elif col == "pct_gain_res":
                labels_by_mode.append("Sequential Reserve Saved")
            else:  # pct_gain_act
                labels_by_mode.append("Sequential Activation Gain")

axes[1, 1].boxplot(data_by_mode, labels=labels_by_mode, showfliers=False)
axes[1, 1].set_title('Contributions by MODE')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.show()
