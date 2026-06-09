import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Data from screenshots
# =========================

freqs = [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]

data = {
    "BERT-tiny SST-2": {
        "freq": freqs,
        "energy": [
            140.26, 119.39, 112.51, 106.37, 97.14,
            96.53, 100.94, 92.45, 100.20, 96.01, 108.87
        ],
        "color": "#3498db"
    },

    "BERT-tiny QNLI": {
        "freq": freqs,
        "energy": [
            727.85, 633.43, 592.36, 559.35, 541.76,
            535.71, 540.68, 536.56, 549.47, 547.55, 604.02
        ],
        "color": "#e67e22"
    },

    "Pythia-14M QNLI": {
        "freq": freqs,
        "energy": [
            2108.76, 2025.87, 2031.26, 2069.65, 2099.04,
            2253.58, 2361.20, 2530.17, 2661.67, 2703.66, 2865.45
        ],
        "color": "#2ecc71"
    },

    "Pythia-14M SST-2": {
        "freq": freqs,
        "energy": [
            992.13, 1023.11, 951.29, 990.92, 1064.72,
            1108.23, 1217.30, 1319.65, 1194.58, 1289.94, 1320.79
        ],
        "color": "#9b59b6"
    },

    "Pythia-31M SST-2": {
        "freq": freqs,
        "energy": [
            1642.99, 1519.51, 1482.93, 1563.06, 1588.81,
            1677.93, 1766.39, 1884.86, 2046.84, 2035.49, 2296.69
        ],
        "color": "#e74c3c"
    }
}

# =========================
# Plot
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=120)

all_energy_values = []

for label, values in data.items():
    df = pd.DataFrame({
        "config_freq_mhz": values["freq"],
        "energy_j": values["energy"]
    })

    all_energy_values.extend(df["energy_j"].tolist())

    # Line
    ax.plot(
        df["config_freq_mhz"],
        df["energy_j"],
        color=values["color"],
        marker="o",
        markersize=5,
        linewidth=2.2,
        label=label
    )

    # Optimal point
    opt_idx = df["energy_j"].idxmin()
    opt_x = df.loc[opt_idx, "config_freq_mhz"]
    opt_y = df.loc[opt_idx, "energy_j"]

    ax.scatter(
        opt_x,
        opt_y,
        color=values["color"],
        marker="*",
        s=260,
        edgecolor="black",
        linewidth=0.6,
        zorder=5
    )

    # Optimal frequency label
    ax.text(
        opt_x,
        opt_y + 65,
        f"{opt_x} MHz",
        color=values["color"],
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="bottom"
    )

# Dummy point for legend: optimal marker
ax.scatter(
    [],
    [],
    color="white",
    edgecolor="black",
    marker="*",
    s=220,
    label="Optimal"
)

# =========================
# Titles and labels
# =========================
ax.set_title(
    "BERT and Pythia Models\nEnergy vs GPU Frequency",
    fontsize=14,
    fontweight="bold",
    pad=10
)

ax.set_xlabel(
    "GPU Frequency (MHz)",
    fontsize=11,
    fontstyle="italic"
)

ax.set_ylabel(
    "Energy (J)",
    fontsize=11
)

# =========================
# Grid and axes styling
# =========================
ax.grid(
    True,
    linestyle="-",
    linewidth=0.8,
    alpha=0.25
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#777777")
ax.spines["bottom"].set_color("#777777")

ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(axis="both", labelsize=9, colors="#333333")

# =========================
# Axis limits
# =========================
ax.set_xticks(freqs)
ax.set_xlim(min(freqs) - 60, max(freqs) + 50)

y_min_raw = min(all_energy_values)
y_max_raw = max(all_energy_values)

y_min = math.floor((y_min_raw - 150) / 100) * 100
y_max = math.ceil((y_max_raw + 200) / 100) * 100

ax.set_ylim(y_min, y_max)

# Avoid scientific notation
ax.ticklabel_format(style="plain", axis="y")

# =========================
# Legend under graph
# =========================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=3,
    frameon=True,
    fontsize=8.5
)

plt.tight_layout()
plt.show()
