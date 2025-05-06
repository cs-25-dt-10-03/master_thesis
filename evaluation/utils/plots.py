import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = '../results/summary.csv'

df = pd.read_csv(RESULTS_DIR)


# Create a descriptive scenario label
df['scenario'] = df.apply(lambda row: f"{row['MODE']}_{'Res' if row['RUN_RESERVE'] else 'NoRes'}_{'Act' if row['RUN_ACTIVATION'] else 'NoAct'}", axis=1)

# Plot 1: Savings metrics by scenario
# plt.figure()
# metrics = ['pct_saved_spot', 'pct_gain_res', 'pct_gain_act', 'pct_total_saved']
# df.set_index('scenario')[metrics].plot(kind='bar')
# plt.ylabel('Percentage (%)')
# plt.title('Savings Metrics by Scenario')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()



# pivot = df.pivot(index='scenario', columns='TIME_RES', values='runtime_scheduling')

# # Plot grouped bars
# plt.figure()
# pivot.plot(kind='bar')
# plt.ylabel('Scheduling Runtime (s)')
# plt.title('Scheduling Runtime by Scenario & Time Resolution')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

grouped = df.groupby(['NUM_EVS', 'TIME_RES'])[['runtime_scheduling', 'runtime_aggregation']].mean().reset_index()


sched_pivot = grouped.pivot(index='NUM_EVS', columns='TIME_RES', values='runtime_scheduling')

# Plot: Scheduling Runtime vs Number of FlexOffers
plt.figure()
sched_pivot.plot(marker='o')
plt.xlabel('Number of FlexOffers')
plt.ylabel('Scheduling Runtime (s)')
plt.title('Scheduling Runtime vs Number of FlexOffers')
plt.xticks(rotation=0)
plt.tight_layout()

# Pivot for aggregation runtime
agg_pivot = grouped.pivot(index='NUM_EVS', columns='TIME_RES', values='runtime_aggregation')

# Plot: Aggregation Runtime vs Number of FlexOffers
plt.figure()
agg_pivot.plot(marker='o')
plt.xlabel('Number of FlexOffers')
plt.ylabel('Aggregation Runtime (s)')
plt.title('Aggregation Runtime vs Number of FlexOffers')
plt.xticks(rotation=0)
plt.tight_layout()



# Plot 3: Runtime vs Total Savings scatter
# plt.figure()
# plt.scatter(df['runtime_scheduling'], df['pct_total_saved'])
# for _, row in df.iterrows():
#     plt.annotate(row['scenario'], (row['runtime_scheduling'], row['pct_total_saved']),
#                  textcoords="offset points", xytext=(0,5), ha='center')
# plt.xlabel('Scheduling Runtime (s)')
# plt.ylabel('Total Savings (%)')
# plt.title('Scheduling Runtime vs Total Savings')
# plt.tight_layout()

# # Plot 4: Percentage of Optimal by scenario
# plt.figure()
# plt.plot(df['scenario'], df['pct_of_optimal'], marker='o')
# plt.ylabel('Pct of Optimal (%)')
# plt.title('Pct of Optimal by Scenario')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()

plt.show()