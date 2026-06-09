import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Data from screenshots
# =========================

freqs = [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]

pythia31m_qnli = {
    "config_freq_mhz": freqs,
    "energy_j": [
        3851.11, 3577.59, 3426.29, 3437.27, 3402.86,
        3654.64, 3724.19, 3935.32, 4052.86, 4301.46, 4523.76
    ]
}

pythia70m_qnli = {
    "config_freq_mhz": freqs,
    "energy_j": [
        6253.39, 5805.49, 5466.03, 5418.65, 5377.98,
        5458.80, 5612.99, 5975.11, 6290.24, 6502.40, 7167.90
    ]
}

pythia70m_sst2 = {
    "config_freq_mhz": freqs,
    "energy_j": [
        1894.76, 1723.90, 1643.96, 1682.73, 1716.79,
        1860.43, 1948.80, 2147.00, 2305.11, 2481.36, 2517.25
    ]
}

df_31_qnli = pd.DataFrame(pythia31m_qnli)
df_70_qnli = pd.DataFrame(pythia70m_qnli)
df_70_sst2 = pd.DataFrame(pythia70m_sst2)

# =========================
# Find optimal points
# =========================
opt_31_qnli_idx = df_31_qnli["energy_j"].idxmin()
opt_31_qnli_x = df_31_qnli.loc[opt_31_qnli_idx, "config_freq_mhz"]
opt_31_qnli_y = df_31_qnli.loc[opt_31_qnli_idx, "energy_j"]

opt_70_qnli_idx = df_70_qnli["energy_j"].idxmin()
opt_70_qnli_x = df_70_qnli.loc[opt_70_qnli_idx, "config_freq_mhz"]
opt_70_qnli_y = df_70_qnli.loc[opt_70_qnli_idx, "energy_j"]

opt_70_sst2_idx = df_70_sst2["energy_j"].idxmin()
opt_70_sst2_x = df_70_sst2.loc[opt_70_sst2_idx, "config_freq_mhz"]
opt_70_sst2_y = df_70_sst2.loc[opt_70_sst2_idx, "energy_j"]

# =========================
# Plot
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=120)

# Pythia-31M QNLI
ax.plot(
    df_31_qnli["config_freq_mhz"],
    df_31_qnli["energy_j"],
    color="#3498db",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="Pythia-31M QNLI"
)

# Pythia-70M QNLI
ax.plot(
    df_70_qnli["config_freq_mhz"],
    df_70_qnli["energy_j"],
    color="#e67e22",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="Pythia-70M QNLI"
)

# Pythia-70M SST-2
ax.plot(
    df_70_sst2["config_freq_mhz"],
    df_70_sst2["energy_j"],
    color="#2ecc71",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="Pythia-70M SST-2"
)

# =========================
# Optimal points
# =========================
ax.scatter(
    opt_31_qnli_x,
    opt_31_qnli_y,
    color="#3498db",
    marker="*",
    s=260,
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
    label="31M QNLI Optimal"
)

ax.scatter(
    opt_70_qnli_x,
    opt_70_qnli_y,
    color="#e67e22",
    marker="*",
    s=260,
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
    label="70M QNLI Optimal"
)

ax.scatter(
    opt_70_sst2_x,
    opt_70_sst2_y,
    color="#2ecc71",
    marker="*",
    s=260,
    edgecolor="black",
    linewidth=0.6,
    zorder=5,
    label="70M SST-2 Optimal"
)

# =========================
# Optimal labels
# =========================
ax.text(
    opt_31_qnli_x,
    opt_31_qnli_y - 180,
    f"{opt_31_qnli_x} MHz",
    color="#3498db",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax.text(
    opt_70_qnli_x,
    opt_70_qnli_y - 220,
    f"{opt_70_qnli_x} MHz",
    color="#e67e22",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax.text(
    opt_70_sst2_x,
    opt_70_sst2_y + 180,
    f"{opt_70_sst2_x} MHz",
    color="#27ae60",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="bottom"
)

# =========================
# Titles and labels
# =========================
ax.set_title(
    "Pythia Models\nEnergy vs GPU Frequency",
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

all_energy_values = (
    df_31_qnli["energy_j"].tolist()
    + df_70_qnli["energy_j"].tolist()
    + df_70_sst2["energy_j"].tolist()
)

y_min_raw = min(all_energy_values)
y_max_raw = max(all_energy_values)

y_min = math.floor((y_min_raw - 300) / 500) * 500
y_max = math.ceil((y_max_raw + 300) / 500) * 500

ax.set_ylim(y_min, y_max)

# Avoid scientific notation
ax.ticklabel_format(style="plain", axis="y")

# =========================
# Legend under graph
# =========================
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2,
    frameon=True,
    fontsize=8.5
)

plt.tight_layout()
plt.show()
