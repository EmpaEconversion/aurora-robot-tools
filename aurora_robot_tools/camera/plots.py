

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    ax.add_artist(plt.Circle((cx1, cy1), r1, color="blue", alpha=0.5, label=f"Component {component1}"))
    ax.add_artist(plt.Circle((cx2, cy2), r2, color="red", alpha=0.5, label=f"Component {component2}"))
    ax.plot(cx1, cy1, "bo", markersize = 2)  # Center for component 1
    ax.plot(cx2, cy2, "ro", markersize = 2)  # Center for component 2

# Add legend
handles = [
    plt.Line2D([0], [0], color="blue", marker="o", linestyle="None", label="Anode"),
    plt.Line2D([0], [0], color="red", marker="o", linestyle="None", label="Cathode"),
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

