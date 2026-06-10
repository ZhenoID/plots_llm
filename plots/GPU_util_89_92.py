import pandas as pd
import matplotlib.pyplot as plt
import math

# =========================
# Data from screenshot
# =========================

# SST-2 manual sweep
sst2_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        31515.30, 28564.77, 26142.12, 25516.60, 25221.15,
        25800.00, 26104.58, 27550.02, 28899.26, 30970.72, 32958.88
    ]
}

# QNLI manual sweep
qnli_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        26564.12, 23590.82, 21458.72, 21166.94, 21038.73,
        21034.89, 21791.35, 22903.66, 24153.37, 26006.20, 27490.58
    ]
}

# SmolLM-135M SST-2 manual sweep from screenshot
smollm_sst2_data = {
    "config_freq_mhz": [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    "energy_j": [
        16521.67, 14435.58, 12979.18, 12749.46, 13024.01,
        13104.29, 13471.14, 14256.86, 15321.55, 16134.66, 17128.47
    ]
}

# Mode0 values from screenshots
sst2_mode0_energy = 32713.40
qnli_mode0_energy = 20071.39
smollm_sst2_mode0_energy = 17168.90

sst2_df = pd.DataFrame(sst2_data)
qnli_df = pd.DataFrame(qnli_data)
smollm_sst2_df = pd.DataFrame(smollm_sst2_data)

# =========================
# Find optimal points
# =========================
sst2_opt_idx = sst2_df["energy_j"].idxmin()
sst2_opt_x = sst2_df.loc[sst2_opt_idx, "config_freq_mhz"]
sst2_opt_y = sst2_df.loc[sst2_opt_idx, "energy_j"]

qnli_opt_idx = qnli_df["energy_j"].idxmin()
qnli_opt_x = qnli_df.loc[qnli_opt_idx, "config_freq_mhz"]
qnli_opt_y = qnli_df.loc[qnli_opt_idx, "energy_j"]

smollm_sst2_opt_idx = smollm_sst2_df["energy_j"].idxmin()
smollm_sst2_opt_x = smollm_sst2_df.loc[smollm_sst2_opt_idx, "config_freq_mhz"]
smollm_sst2_opt_y = smollm_sst2_df.loc[smollm_sst2_opt_idx, "energy_j"]

# =========================
# Plot
# =========================
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=120)

# SST-2 line
ax.plot(
    sst2_df["config_freq_mhz"],
    sst2_df["energy_j"],
    color="#3498db",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="BERT SST-2"
)

# QNLI line
ax.plot(
    qnli_df["config_freq_mhz"],
    qnli_df["energy_j"],
    color="#e67e22",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="BERT QNLI"
)

# SmolLM-135M SST-2 line
ax.plot(
    smollm_sst2_df["config_freq_mhz"],
    smollm_sst2_df["energy_j"],
    color="#9b59b6",
    marker="o",
    markersize=5,
    linewidth=2.2,
    label="SmolLM-135M SST-2"
)

# SST-2 optimal point
ax.scatter(
    sst2_opt_x,
    sst2_opt_y,
    color="#2ecc71",
    marker="*",
    s=300,
    zorder=5,
    label="BERT SST-2 Optimal"
)

# QNLI optimal point
ax.scatter(
    qnli_opt_x,
    qnli_opt_y,
    color="#e74c3c",
    marker="*",
    s=300,
    zorder=5,
    label="BERT QNLI Optimal"
)

# SmolLM-135M SST-2 optimal point
ax.scatter(
    smollm_sst2_opt_x,
    smollm_sst2_opt_y,
    color="#f1c40f",
    marker="*",
    s=300,
    zorder=5,
    label="SmolLM SST-2 Optimal"
)

# Optimal labels
ax.text(
    sst2_opt_x,
    sst2_opt_y + 450,
    f"{sst2_opt_x} MHz",
    color="#27ae60",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="bottom"
)

ax.text(
    qnli_opt_x,
    qnli_opt_y - 450,
    f"{qnli_opt_x} MHz",
    color="#c0392b",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

ax.text(
    smollm_sst2_opt_x,
    smollm_sst2_opt_y - 450,
    f"{smollm_sst2_opt_x} MHz",
    color="#b7950b",
    fontsize=9,
    fontweight="bold",
    ha="center",
    va="top"
)

# Mode0 dashed lines
ax.axhline(
    y=sst2_mode0_energy,
    color="#3498db",
    linestyle="--",
    linewidth=1.5,
    label="BERT SST-2 MAXN (Mode0)"
)

ax.axhline(
    y=qnli_mode0_energy,
    color="#e67e22",
    linestyle="--",
    linewidth=1.5,
    label="BERT QNLI MAXN (Mode0)"
)

ax.axhline(
    y=smollm_sst2_mode0_energy,
    color="#9b59b6",
    linestyle="--",
    linewidth=1.5,
    label="SmolLM SST-2 MAXN (Mode0)"
)

# =========================
# Titles and labels
# =========================
ax.set_title(
    "Energy vs GPU Frequency",
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

# Show only existing frequencies
all_x = sorted(
    set(sst2_df["config_freq_mhz"])
    .union(set(qnli_df["config_freq_mhz"]))
    .union(set(smollm_sst2_df["config_freq_mhz"]))
)

ax.set_xticks(all_x)

# Axis limits
ax.set_xlim(min(all_x) - 60, max(all_x) + 50)

y_min_raw = min(
    sst2_df["energy_j"].min(),
    qnli_df["energy_j"].min(),
    smollm_sst2_df["energy_j"].min(),
    sst2_mode0_energy,
    qnli_mode0_energy,
    smollm_sst2_mode0_energy
)

y_max_raw = max(
    sst2_df["energy_j"].max(),
    qnli_df["energy_j"].max(),
    smollm_sst2_df["energy_j"].max(),
    sst2_mode0_energy,
    qnli_mode0_energy,
    smollm_sst2_mode0_energy
)

y_min = math.floor((y_min_raw - 1000) / 1000) * 1000
y_max = math.ceil((y_max_raw + 1000) / 1000) * 1000

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
    fontsize=8.0
)

plt.tight_layout()
plt.show()
