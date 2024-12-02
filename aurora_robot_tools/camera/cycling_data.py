""" Lina Scholz

Script to alanyse/plot performance data.
"""

import math
import os
import json
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px

#%%

cycling_data = r"G:\Limit\Lina Scholz\Cell Data\batch.lisc_gen14.json"

keys = ['Sample ID', 'Cycle', 'Charge capacity (mAh)', 'Discharge capacity (mAh)', 'Efficiency (%)',
        'Specific charge capacity (mAh/g)', 'Specific discharge capacity (mAh/g)', 'Normalised discharge capacity (%)',
        'Normalised discharge energy (%)', 'Charge average voltage (V)', 'Discharge average voltage (V)', 'Delta V (V)',
        'Charge average current (A)', 'Discharge average current (A)', 'Charge energy (mWh)', 'Discharge energy (mWh)',
        'Max voltage (V)', 'Formation C', 'Cycle C', 'Actual N:P ratio', 'Anode type', 'Cathode type',
        'Anode active material mass (mg)', 'Cathode active material mass (mg)', 'Electrolyte name',
        'Electrolyte description', 'Electrolyte amount (uL)', 'Rack position', 'First formation efficiency (%)',
        'First formation specific discharge capacity (mAh/g)', 'Initial specific discharge capacity (mAh/g)',
        'Initial efficiency (%)', 'Capacity loss (%)', 'Last specific discharge capacity (mAh/g)',
        'Last efficiency (%)', 'Formation average voltage (V)', 'Formation average current (A)', 'Initial delta V (V)',
        'Cycles to 95% capacity', 'Cycles to 90% capacity', 'Cycles to 85% capacity', 'Cycles to 80% capacity',
        'Cycles to 75% capacity', 'Cycles to 70% capacity', 'Cycles to 60% capacity', 'Cycles to 50% capacity',
        'Cycles to 95% energy', 'Cycles to 90% energy', 'Cycles to 85% energy', 'Cycles to 80% energy',
        'Cycles to 75% energy', 'Cycles to 70% energy', 'Cycles to 60% energy', 'Cycles to 50% energy', 'Run ID',
        'Electrolyte to press (s)', 'Electrolyte to electrode (s)', 'Electrode to protection (s)', 'Press to protection (s)']

# Open and load the JSON file
with open(cycling_data, 'r') as file:
    json_data = json.load(file)
    # Extract the data associated with the key "data"
    if "data" in json_data:
        cell_data = json_data["data"]

cells = {}
for cell in cell_data:
    data = pd.DataFrame()
    for string in keys:
        data[string] = None # initialize columns
    number = int(cell["Sample ID"].split("_")[-1])
    for i in range(len(keys)):
        data_points = cell[keys[i]]
        data[keys[i]] = data_points
    cells[number] = data

cells = collections.OrderedDict(sorted(cells.items()))

#%%

fig, ax = plt.subplots()
x = "Cycle"
y = "Specific discharge capacity (mAh/g)"
# Get the viridis colormap
cmap = cm.viridis
num_cells = len(cells)
colors = [cmap(i / num_cells) for i in range(num_cells)]  # Generate a list of colors
# Plot discharge capacity vs. cycle number
for (key, value), color in zip(cells.items(), colors):
    ax.scatter(value[x], value[y], label=f"Cell {key}", color=color, s=16)
# Label the axes
ax.set_xlabel(f"{x}", fontsize=14)
ax.set_ylabel(f"{y}", fontsize=14)
ax.set_xlim(4, 230)
# Position the legend outside the plot on top
ax.legend(loc='upper center', bbox_to_anchor=(0.75, 0.975), ncol=4, fontsize=10)  # Adjust ncol for more columns
plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
x = "Cell Number"
y = "Cycles to 70% capacity"
# Get the viridis colormap
cmap = cm.viridis
num_cells = len(cells)
colors = [cmap(i / num_cells) for i in range(num_cells)]  # Generate a list of colors
# Plot discharge capacity vs. cycle number
for (key, value), color in zip(cells.items(), colors):
    ax.scatter(key, value[y][0], label=f"Cell {key}", color="black", s=16)
# Label the axes
ax.set_xlabel(f"{x}", fontsize=14)
ax.set_ylabel(f"{y}", fontsize=14)
#  ax.set_xlim(4, 230)
# Position the legend outside the plot on top
# ax.legend(loc='upper center', bbox_to_anchor=(0.75, 0.95), ncol=4, fontsize=10)  # Adjust ncol for more columns
plt.tight_layout()
plt.show()

#%%

# Create a combined DataFrame with Cycle, Specific discharge capacity, and Cell info for hover
all_data = []
for key, value in cells.items():
    value['Cell'] = key  # Add the cell identifier to the dataframe
    all_data.append(value)

# Combine the data from all cells into one DataFrame
df_combined = pd.concat(all_data)

# Create a Plotly scatter plot
fig = px.scatter(df_combined, 
                 x="Cycle", 
                 y="Specific discharge capacity (mAh/g)", 
                 color="Cell",  # Use 'Cell' to assign colors based on the key
                 labels={"Cycle": "Cycle", "Specific discharge capacity (mAh/g)": "Specific discharge capacity (mAh/g)"},
                 hover_data=["Cell"],  # Show the 'Cell' key on hover
                 color_continuous_scale="viridis",  # Use the viridis colormap
                 title="Discharge Capacity vs. Cycle Number")

# Update layout to move the legend outside the plot
fig.update_layout(
    legend_title="Cell",
    legend=dict(
        x=1.05,  # Place legend outside
        y=1,     # Position it at the top
        traceorder="normal",
        orientation="h",  # Horizontal legend
        font=dict(size=10)
    ),
    xaxis=dict(range=[4, 230])  # Set x-axis limits
)

# Show the plot
fig.show()
