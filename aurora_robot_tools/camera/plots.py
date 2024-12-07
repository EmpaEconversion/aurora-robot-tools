

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Get Seaborn's default blue and red
sns_palette = sns.color_palette("deep")
seaborn_blue = sns_palette[0]  # First color in the palette (blue)
seaborn_red = sns_palette[3]   # Fourth color in the palette (red)

#%%

df = pd.read_excel("C:/lisc_gen14/data/alignment_manual_final.xlsx")
radii = [10, 10, 7.5, 7.5, 8, 8, 7, 8, 7.5, 10, 10]

# Define the components to plot
component1 = 2
component2 = 6

# Sort data by press and cell number
df = df.sort_values(by=["cell"]).reset_index(drop=True)

# Calculate grid positions
df["grid_x"] = df["press"]
df["grid_y"] = (df["cell"] - 1) // 6 + 1  # Batch positions

# Create the plot
fig, ax = plt.subplots(figsize=(8, 8), layout="tight")

offset = 20  # Spacing between grid centers

for _, row in df.iterrows():
    # Extract coordinates and radii for the two components
    x1, y1, r1 = row[f"x{component1}"], -row[f"y{component1}"], radii[component1]
    x2, y2, r2 = row[f"x{component2}"], -row[f"y{component2}"], radii[component2]

    # Translate grid positions to the reference center
    grid_center_x = row["grid_x"] * offset
    grid_center_y = row["grid_y"] * offset

    # Calculate actual positions by adding the deviations
    cx1, cy1 = grid_center_x + x1, grid_center_y + y1
    cx2, cy2 = grid_center_x + x2, grid_center_y + y2

    # Plot circles and their centers
    ax.add_artist(plt.Circle((cx1, cy1), r1, color="purple", alpha=0.6, label=f"Component {component1}"))
    ax.add_artist(plt.Circle((cx2, cy2), r2, color="darkcyan", alpha=0.6, label=f"Component {component2}"))
    ax.plot(cx1, cy1, "o", color="magenta", markersize=2)  # Center for component 1 (black)
    ax.plot(cx2, cy2, "o", color="cyan", markersize=2)  # Center for component 2 (purple)

# Add legend
handles = [
    plt.Line2D([0], [0], color="purple", marker="o", linestyle="None", label="Anode"),
    plt.Line2D([0], [0], color="darkcyan", marker="o", linestyle="None", label="Cathode"),
]
ax.legend(handles=handles, loc="lower left", fontsize = 14)

# Set axis limits and labels
ax.set_xlim(0, (df["grid_x"].max() + 1) * offset)
ax.set_ylim(0, (df["grid_y"].max() + 1) * offset)
ax.set_xlabel("Pressing Tool Position", fontsize=16)
ax.set_ylabel("Production Batch", fontsize=16)

# Adjust grid lines and ticks
ax.set_xticks(df["grid_x"].unique() * offset)
ax.set_yticks(np.arange(1, df["grid_y"].max() + 1) * offset)
ax.grid(True, linestyle="--", alpha=0.5)

# Relabel the x-axis
xtick_positions = np.arange(offset, (df["grid_x"].max() + 1) * offset, offset)
ax.set_xticks(xtick_positions)
ax.set_xticklabels(np.arange(1, len(xtick_positions) + 1))

# Relabel the y-axis
ytick_positions = np.arange(offset, (df["grid_y"].max() + 1) * offset, offset)
ax.set_yticks(ytick_positions)
ax.set_yticklabels(np.arange(1, len(ytick_positions) + 1))

# Sizes
ax.tick_params(axis="both", labelsize=14)

plt.show()

#%%

performance = pd.read_excel("C:/lisc_gen14/data/performance_final.xlsx")
# values: d26, electrodes_to_press, Fade rate 5-150 cycles (%/cycle),
# Initial specific discharge capacity (mAh/g), Specific discharge capacity 180th (mAh/g),
# Cycles to 70% capacity, Initial efficiency (%)

plots2 = False
plots3 = True

if plots2:
    y_list = ["Fade rate 5-150 cycles (%/cycle)",  "Initial specific discharge capacity (mAh/g)",
        "Specific discharge capacity 180th (mAh/g)", "Cycles to 70% capacity", "Initial efficiency (%)",
        'First formation efficiency (%)', 'First formation specific discharge capacity (mAh/g)']

    x = "d26"
    for y in y_list:
        fig, ax = plt.subplots(figsize=(8, 8), layout="tight")
        ax.scatter(performance[x], performance[y])
        ax.set_xlabel("electrode alignment")
        ax.set_ylabel(y)
        #plt.show()

    x = "electrodes_to_press"
    for y in y_list:
        fig, ax = plt.subplots(figsize=(8, 8), layout="tight")
        ax.scatter(performance[x], performance[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # plt.show()

if plots3:
    x = "press"
    for y in ["d26", "z8", "z2", "z6"]:
        fig, ax = plt.subplots(figsize=(8, 8), layout="tight")
        ax.scatter(performance[x], performance[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plt.show()

#%%

plots4 = False

if plots4:
    x = "d26"
    ys = ["Initial specific discharge capacity (mAh/g)", "Initial efficiency (%)"]

    fig, ax = plt.subplots(1, len(ys), figsize=(8, 4), constrained_layout=True)
    for i, y in enumerate(ys):
        ax[i].scatter(performance[x], performance[y], color=seaborn_blue, s=20)
        ax[i].set_xlabel("Electrode alignment [mm]", fontsize = 14)
        ax[i].set_ylabel(y, fontsize = 14)
        ax[i].tick_params(labelsize=12)

    plt.show()

    fig, axes = plt.subplots(1, len(ys), figsize=(8, 4), constrained_layout=True)
    for i, y in enumerate(ys):
        # Scatter plot with regression line
        sns.regplot(x=performance[x], y=performance[y], ax=axes[i], scatter_kws={"s": 20}, line_kws={"color": "blue"})

        # Calculate r and p values
        r, p = stats.pearsonr(performance[x], performance[y])

        # Annotate r and p values
        axes[i].text(
            0.05, 0.95, f"r={r:.2f}, p={p:.1e}", 
            transform=axes[i].transAxes, 
            ha="left", va="top", fontsize=10
        )
        # Labels and ticks
        axes[i].set_xlabel("Electrode alignment [mm]", fontsize=12)
        axes[i].set_ylabel(y, fontsize=12)
        axes[i].tick_params(labelsize=10)

    plt.show()

# %%
