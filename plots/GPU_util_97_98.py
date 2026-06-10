import pandas as pd
import matplotlib.pyplot as plt
import math

# QNLI
qnli_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        693591.48, 596669.90, 545930.79, 536503.90, 539312.18,
        549624.25, 572217.63, 610475.85, 641276.34, 698390.84, 737041.01
    ]
}
qnli_mode0_energy = 552288.86

# SST-2
sst2_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        55292.41, 47512.24, 43025.92, 41044.05, 40927.33,
        40981.54, 42053.11, 44378.81, 46298.01, 49986.88, 52475.58
    ]
}
sst2_mode0_energy = 52317.17

qnli_df = pd.DataFrame(qnli_data)
sst2_df = pd.DataFrame(sst2_data)

# =========================
# Find optimal points
# =========================
qnli_opt_idx = qnli_df["energy_j"].idxmin()
qnli_opt_x = qnli_df.loc[qnli_opt_idx, "config_freq_mhz"]
qnli_opt_y = qnli_df.loc[qnli_opt_idx, "energy_j"]

sst2_opt_idx = sst2_df["energy_j"].idxmin()
sst2_opt_x = sst2_df.loc[sst2_opt_idx, "config_freq_mhz"]
sst2_opt_y = sst2_df.loc[sst2_opt_idx, "energy_j"]

# =========================
# Plot
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=120)

# QNLI line
ax.plot(
    qnli_df["config_freq_mhz"],
    qnli_df["energy_j"],
    color="#3498db",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="QNLI"
)

# SST-2 line
ax.plot(
    sst2_df["config_freq_mhz"],
    sst2_df["energy_j"],
    color="#e67e22",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="SST-2"
)

# Optimal points
ax.scatter(
    qnli_opt_x, qnli_opt_y,
    color="#2ecc71",
    marker="*",
    s=260,
    zorder=5,
    label="QNLI Optimal"
)

ax.scatter(
    sst2_opt_x, sst2_opt_y,
    color="#e74c3c",
    marker="*",
    s=260,
    zorder=5,
    label="SST-2 Optimal"
)

# Optimal labels
ax.text(
    qnli_opt_x,
    qnli_opt_y - 18000,
    f"{qnli_opt_x} MHz",
    color="#27ae60",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax.text(
    sst2_opt_x,
    sst2_opt_y + 12000,
    f"{sst2_opt_x} MHz",
    color="#c0392b",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="bottom"
)

# Mode0 dashed lines
ax.axhline(
    y=qnli_mode0_energy,
    color="#3498db",
    linestyle="--",
    linewidth=1.5,
    label="QNLI MAXN (Mode0)"
)

ax.axhline(
    y=sst2_mode0_energy,
    color="#e67e22",
    linestyle="--",
    linewidth=1.5,
    label="SST-2 MAXN (Mode0)"
)

# =========================
# Titles and labels
# =========================
ax.set_title(
    "DeBERTa-xlarge\nEnergy vs GPU Frequency",
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
# Grid and style
# =========================
ax.grid(True, linestyle="-", linewidth=0.8, alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.spines["left"].set_color("#777777")
ax.spines["bottom"].set_color("#777777")
ax.spines["left"].set_linewidth(1.2)
ax.spines["bottom"].set_linewidth(1.2)

ax.tick_params(axis="both", labelsize=9, colors="#333333")

# Show only existing frequencies
all_x = sorted(set(qnli_df["config_freq_mhz"]).union(set(sst2_df["config_freq_mhz"])))
ax.set_xticks(all_x)

# Avoid scientific notation
ax.ticklabel_format(style="plain", axis="y")

# Axis limits
x_min = min(all_x) - 60
x_max = max(all_x) + 50
ax.set_xlim(x_min, x_max)

y_min_raw = min(sst2_df["energy_j"].min(), sst2_mode0_energy, qnli_df["energy_j"].min(), qnli_mode0_energy)
y_max_raw = max(sst2_df["energy_j"].max(), sst2_mode0_energy, qnli_df["energy_j"].max(), qnli_mode0_energy)

y_min = math.floor((y_min_raw - 30000) / 10000) * 10000
y_max = math.ceil((y_max_raw + 30000) / 10000) * 10000
ax.set_ylim(y_min, y_max)

# Legend
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.23),
    ncol=3,
    frameon=True,
    fontsize=9
)

plt.tight_layout()
plt.show()
