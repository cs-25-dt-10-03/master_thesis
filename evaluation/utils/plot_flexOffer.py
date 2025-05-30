import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_flexoffer(fo, spot_prices=None, resolution_seconds=3600, time_fmt="%H:%M"):
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    t0 = fo.get_est()
    profile = fo.get_profile()
    duration = fo.get_duration()
    sched0 = int((fo.get_scheduled_start_time() - t0) // resolution_seconds)
    sched = fo.get_scheduled_allocation()
    e_min = round(fo.get_min_overall_alloc())
    e_max = round(fo.get_max_overall_alloc())

    est_slot = 0
    lst_slot = int((fo.get_lst() - t0) // resolution_seconds)
    et_slot  = int((fo.get_et() - t0) // resolution_seconds)
    et_slot -= 1

    full_slots = np.arange(0, et_slot +1)  # show all hours until end time
    bar_width = 0.45
    fig, ax = plt.subplots(figsize=(12, 4))

    # Determine y-limit max
    y_max = max((ts.max_power for ts in profile), default=1)

    # Draw energy bars (if defined), otherwise blank
    for j in full_slots:
        if j < len(profile):
            ts = profile[j]
            ax.bar(j - bar_width/2, ts.min_power, width=bar_width, bottom=0,
                   color='#4682b4', edgecolor='blue', align='edge', zorder=1)
            ax.bar(j - bar_width/2, ts.max_power - ts.min_power, width=bar_width,
                   bottom=ts.min_power, color='#add8e6', edgecolor='blue', align='edge', zorder=1)
        # Always write e_i
        ax.text(j, y_max * 1.05, f"$e_{{{j+1}}}$", ha='center', va='bottom', fontsize=9)

    # Scheduled allocation (even if bar is missing)
    for j, p in enumerate(sched):
        x0 = sched0 + j
        ax.hlines(p, x0 - bar_width/2, x0 + bar_width/2,
                  linestyles='-', linewidth=1.5, color='black', zorder=3)

    # Time labels
    times = [datetime.fromtimestamp(t0 + i * resolution_seconds).strftime(time_fmt) for i in full_slots]
    labels = times
    if spot_prices is not None:
        labels = []
        for i in full_slots:
            ts = datetime.fromtimestamp(t0 + i * resolution_seconds)
            price = spot_prices.get(ts, "N/A") if hasattr(spot_prices, 'get') else spot_prices.loc.get(ts, "N/A")
            label = f"{times[i]}\n{int(price)} DKK" if isinstance(price, (int, float)) else f"{times[i]}\nN/A"
            labels.append(label)

    ax.set_xticks(full_slots)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)

    # Event lines + labels ABOVE bars
    event_lines = [
        (est_slot - bar_width/2, "Earliest Start", "green", y_max * 1.20),
        (lst_slot - bar_width/2, "Latest Start",   "orange", y_max * 1.18),
        ((et_slot - bar_width/2) + 0.5,         "End Time",       "red",    y_max * 1.20)
    ]
    for x, label, color, y in event_lines:
        ax.axvline(x, color=color, linestyle='--', linewidth=1.2, zorder=2)
        ax.text(x, y, label, ha='left', va='bottom', fontsize=8, color=color)

    # Titles and limits
    ax.set_xlim(-0.5, et_slot + 0.5)
    ax.set_ylim(0, y_max * 1.3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (kWh)")
    ax.set_title(
        rf"${e_min}\,\mathrm{{kWh}} \leq e_1 + \dots + e_{len(profile) + (lst_slot - est_slot)} \leq {e_max}\,\mathrm{{kWh}}$",
        fontsize=13, pad=15
    )

    # Styling
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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
    if spot_prices is not None:
        axes[-1].set_xlabel("Slot / Time / Spot price")
    else:
        axes[-1].set_xlabel("Slot / Time")
    plt.tight_layout()
    plt.show()
