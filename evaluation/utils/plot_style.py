import matplotlib.pyplot as plt

# Color scheme
MARKET_COLORS = {
    "spot": "#1f77b4",         # blue
    "reserve": "#ff7f0e",      # orange
    "activation": "#2ca02c",   # green
}

# Mode style (for different scheduling strategies)
MODE_STYLES = {
    "joint": "-",
    "sequential": "--",
    "sequential_reserve_first": ":"
}

MODE_MARKERS = {
    "joint": "o",
    "sequential": "s",
    "sequential_reserve_first": "v"
}

# Alignment style (for different aggregation strategies)
ALIGNMENT_STYLES = {
    "start": "-",
    "balance": "--",
    "balance_fast": ":"
}

ALIGNMENT_MARKERS = {
    "start": "o",
    "balance": "s",
    "balance_fast": "^"
}

# General font and grid style
FONT = {
    "size": 11,
    "family": "DejaVu Sans"
}

plt.rc("font", **FONT)
plt.rc("axes", titlesize=12)
plt.rc("axes", labelsize=11)
plt.rc("legend", fontsize=10)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)
plt.rc("grid", linestyle="--", alpha=0.6)
