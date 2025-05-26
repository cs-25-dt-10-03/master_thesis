import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_flexoffer(fo, spot_prices=None, resolution_seconds=3600, time_fmt="%H:%M"):
    """
    Draw flexible-envelope bars per slice, with slot indices, timestamps, scheduled allocation,
    event markers, and spot prices.
    """
    t0 = fo.get_est()
    to_slot = lambda ts: int((ts - t0) // resolution_seconds)
    t_es = 0
    t_ls = to_slot(fo.get_lst())
    t_le = to_slot(fo.get_et())
    profile = fo.get_profile()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 4))

    # Envelope bars
    for j, ts in enumerate(profile):
        ax.bar(j, ts.min_power, width=1, bottom=0, align='edge',
               color='lightgrey', edgecolor='black', zorder=1)
        ax.bar(j, ts.max_power - ts.min_power, width=1, bottom=ts.min_power,
               align='edge', color='darkgrey', edgecolor='black', zorder=1)

    # Scheduled allocation
    sched0 = to_slot(fo.get_scheduled_start_time())
    sched = fo.get_scheduled_allocation()
    for j, p in enumerate(sched):
        x0 = sched0 + j
        ax.hlines(p, x0, x0 + 1, linestyles=':', linewidth=2, color='black', zorder=3)

    # Event markers
    y_max = max(ts.max_power for ts in profile)
    events = [(t_es, "Earliest Start", "green"), (t_ls, "Latest Start", "orange"), (t_le, "End Time", "red")]
    for x, lbl, col in events:
        ax.axvline(x, color=col, linewidth=1.5, zorder=4)
        ax.text(x, y_max * 1.02, lbl, ha='center', va='bottom', color=col, fontsize=9, zorder=4)

    # Grid and limits
    ax.set_xlim(0, t_le + 0.5)
    ax.set_ylim(0, y_max * 1.15)
    ax.set_xticks(np.arange(t_le + 1))
    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)
    ax.set_axisbelow(False)

    slots = np.arange(t_le + 1)
    times = [
        datetime.fromtimestamp(t0 + i * resolution_seconds).strftime(time_fmt)
        for i in slots
    ]

    price_labels = [""] * len(slots)
    if spot_prices is not None:
        for i in slots:
            # Shift price lookup by +1 slot
            ts = datetime.fromtimestamp(t0 + (i + 1) * resolution_seconds)
            if ts in spot_prices.index:
                price = spot_prices.loc[ts]
                price_labels[i] = f"\n{int(price)} DKK"
            else:
                price_labels[i] = "\nN/A"

    labels = [f"{i}\n{times[i]}{price_labels[i]}" for i in slots]
    ax.set_xticklabels(labels, fontsize=8)
    # Axis labels
    ax.set_xlabel("Slot index / Time / Spot price")
    ax.set_ylabel("Energy (kW)")
    plt.tight_layout()
    plt.show()


def plot_flexoffer_aggregation(fo1, fo2, fo_agg, spot_prices=None, resolution_seconds=3600, time_fmt="%H:%M"):
    """
    Draw side-by-side flexOffers aligned on a common time axis for intuitive offset comparison,
    all sharing the same y-axis scale based on the aggregated maximum.
    """
    # Prepare list of offers
    fos = [fo1, fo2, fo_agg]

    # Determine global earliest and latest timestamps
    global_est = min(fo.get_est() for fo in fos)
    global_et  = max(fo.get_et()  for fo in fos)

    print()

    # Compute global maximum power to unify y-axis across subplots
    global_y_max = max(
        max(ts.max_power for ts in fo.get_profile())
        for fo in fos
    )

    # Function to convert timestamp to slot index relative to global_est
    to_global_slot = lambda ts: int((ts - global_est) // resolution_seconds)

    # Total number of slots for the global timeline
    total_slots = to_global_slot(global_et) + 1

    # Create subplots for each FlexOffer
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 9), sharex=True, sharey=True)
    titles = ["FlexOffer A", "FlexOffer B", "Aggregated FlexOffer (A+B)"]
    colors = ['#9ecae1', '#a1d99b', '#fdae6b']

    for ax, fo, title, color in zip(axes, fos, titles, colors):
        profile = fo.get_profile()
        offset = to_global_slot(fo.get_est())

        # Plot the envelope bars with proper offset
        for j, ts in enumerate(profile):
            x = offset + j
            ax.bar(x, ts.min_power, width=1, bottom=0, align='edge',
                   color='lightgrey', edgecolor='black', zorder=1)
            ax.bar(x, ts.max_power - ts.min_power, width=1, bottom=ts.min_power,
                   color=color, edgecolor='black', align='edge', zorder=1)

        # Plot scheduled allocation
        # sched_start = to_global_slot(fo.get_scheduled_start_time())
        # alloc = fo.get_scheduled_allocation()
        # for j, p in enumerate(alloc):
        #     x0 = sched_start + j
        #     ax.hlines(p, x0, x0 + 1, linestyles='--', linewidth=2, color='black', zorder=2)

        # Plot event markers (Earliest Start, Latest Start, End Time)
        events = [
            ("Earliest Start", fo.get_est(), "green"),
            ("Latest Start",   fo.get_lst(), "orange"),
            ("End Time",       fo.get_et(),  "red")
        ]
        for lbl, t, col in events:
            x = to_global_slot(t)
            ax.axvline(x, color=col, linewidth=1.5, zorder=3)
            ax.text(x, global_y_max * 1.02, lbl, ha='center', va='bottom', color=col, fontsize=8)

        ax.set_ylabel("kW")
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.7, zorder=2)

    # Apply shared y-axis limit across all plots
    for ax in axes:
        ax.set_ylim(0, global_y_max * 1.15)

    # Shared X-axis formatting
    slots = np.arange(total_slots)
    times = [
        datetime.fromtimestamp(global_est + i * resolution_seconds).strftime(time_fmt)
        for i in slots
    ]

    price_labels = [""] * total_slots
    if spot_prices is not None:
        for i in slots:
            ts = datetime.fromtimestamp(global_est + (i + 1) * resolution_seconds)
            if ts in spot_prices.index:
                price_labels[i] = f"\n{int(spot_prices.loc[ts])} DKK"

    labels = [f"{i}\n{times[i]}{price_labels[i]}" for i in slots]
    axes[-1].set_xticks(slots)
    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[-1].set_xlabel("Slot / Time / Spot price")

    plt.tight_layout()
    plt.show()
