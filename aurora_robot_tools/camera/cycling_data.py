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
import plotly.graph_objects as go
import seaborn as sns

#%%

lineplot = False

cycling_data = r"G:\Limit\Lina Scholz\Cell Data\lisc_gen14\batch.lisc_gen14.json"

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

# Get Seaborn's default blue and red
sns_palette = sns.color_palette("deep")
seaborn_blue = sns_palette[0]  # First color in the palette (blue)
seaborn_red = sns_palette[3]   # Fourth color in the palette (red)

fig, ax = plt.subplots(figsize=(10, 8), layout="tight")
x = "Cycle"
y = "Specific discharge capacity (mAh/g)"
# Farben für die Gruppen
color_group1 = seaborn_blue
color_group2 = seaborn_red
# Plotten der Punkte mit farblicher Gruppierung
for key, value in cells.items():
    # if 2 <= key <= 17:  # Gruppe 1
    ax.scatter(value[x], value[y], color=color_group1, s=8, alpha=0.6)
    # elif key >= 18:  # Gruppe 2
        # ax.scatter(value[x], value[y], color=color_group2, s=8, alpha=0.6)
# Achsenbeschriftungen
ax.set_xlabel(f"{x}", fontsize=16)
ax.set_ylabel(f"{y}", fontsize=16)
ax.set_xlim(4, 350)
# Manuelle Legende für die Gruppen
##group_legend_handles = [
    #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_group1, markersize=4, label="normally aligned"),
    #plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_group2, markersize=4, label="misaligned cathode")
#]
ax.tick_params(axis='both', which='major', labelsize=14)  # 'both' adjusts x and y ticks
#ax.legend(handles=group_legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


fig, ax = plt.subplots()
x = "Cell Number"
y = "Cycles to 70% capacity"
# Farben für die Gruppen
color_group1 = seaborn_blue
color_group2 = seaborn_red
# Plotten der Punkte mit farblicher Gruppierung
for key, value in cells.items():
    if 2 <= key <= 17:  # Gruppe 1
        ax.scatter(key, value[y][0], color=color_group1, s=16)
    elif key >= 18:  # Gruppe 2
        ax.scatter(key, value[y][0], color=color_group2, s=16)
# Achsenbeschriftungen
ax.set_xlabel(f"{x}", fontsize=14)
ax.set_ylabel(f"{y}", fontsize=14)
# Manuelle Legende für die Gruppen
group_legend_handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_group1, markersize=8, label="normally aligned"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_group2, markersize=8, label="misaligned on purpose")
]
ax.legend(handles=group_legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2, fontsize=10)
plt.tight_layout()
plt.show()


#%%

# Batches files are also JSON, and contain a list of cycles files
batches_filename = r"G:\Limit\Lina Scholz\Cell Data\lisc_gen14\batch.lisc_gen14.json"
batches_file = json.load(open(batches_filename))
batches_data = batches_file['data']

# read data from NMC622 graphite Eneas Cells
enea = pd.read_excel(r"C:\Users\lisc\Downloads\hist_enea_img.xlsx")
enea["Initial specific discharge capacity (mAh/g)"] = enea["Specific discharge capacity (mAh/g)"]
enea["Sample ID"] = [(i + 1) for i in range(40, 40 + len(enea))]
# Add a 'Group' column to the new data based on the `Sample ID`
enea['Group'] = enea['Sample ID'].apply(
    lambda x: 'normally aligned' if 2 <= x <= 17 else
              ('misaligned cathode' if 18 <= x <= 36 else
               ('reference' if x > 36 else 'Other'))
)

# It is useful to split the lists and non-lists into separate dataframes
# This df is huge and contains all the cycles for every sample
batches_list_df = pd.concat([pd.DataFrame(d).iloc[:300] for d in batches_data]).reset_index(drop=True)

# Create a new column for the group based on the Sample ID
batches_list_df['Group'] = batches_list_df['Sample ID'].str[-2:].astype(int)  # Extract the last two digits
batches_list_df['Group'] = batches_list_df['Group'].apply(lambda x: 'normally aligned' if 2 <= x <= 17 else ('misaligned cathode' if 18 <= x <= 36 else 'Other'))

# Filter out 'Other' groups if needed
batches_list_df = batches_list_df[batches_list_df['Group'] != 'Other']

# Drop duplicates in the 'Initial specific discharge capacity (mAh/g)' column
batches_list_df_unique = batches_list_df.drop_duplicates(subset="Initial specific discharge capacity (mAh/g)")
# Combine the new data with the existing DataFrame
combined_df = pd.concat([batches_list_df_unique, enea])

# x over y
single_cell_values = batches_list_df.drop_duplicates(subset="Sample ID")
x_value = "Sample ID"
y_value = "Initial efficiency (%)"
fig, ax = plt.subplots(layout="tight")
ax.scatter(single_cell_values[x_value], single_cell_values[y_value])
ax.set_xlabel(x_value)
ax.set_xticklabels(ax.get_xticks(), rotation = 90)
ax.set_ylabel(y_value)
plt.show()

# x over y
single_cell_values = batches_list_df.drop_duplicates(subset="Sample ID")
x_value = "Sample ID"
y_value = "Initial specific discharge capacity (mAh/g)"
fig, ax = plt.subplots(layout="tight")
ax.scatter(single_cell_values[x_value], single_cell_values[y_value])
ax.set_xlabel(x_value)
ax.set_xticklabels(ax.get_xticks(), rotation = 90)
ax.set_ylabel(y_value)
plt.show()

# Create a custom color palette for the groups
palette = {'normally aligned': 'blue', 'misaligned cathode': 'red', 'reference': 'green'}

# Create the sns lineplot
if lineplot:
    sns.lineplot(
        data=batches_list_df,
        x="Cycle",
        y="Specific discharge capacity (mAh/g)",
        hue='Group',           # Use Group for coloring the lines
        palette=palette,       # Specify the colors for each group
        style='Group',         # This will differentiate the groups by line style
        markers=True           # Optional, to add markers to the lines
    )
    plt.xlim(0, 350)
    plt.show()

# Create the histogram with eneas data
sns.histplot(
    data=combined_df,
    x="Initial specific discharge capacity (mAh/g)",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("Initial Specific Discharge Capacity (mAh/g)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Create the histogram with Initial specific discharge capacity (mAh/g) // First formation specific discharge capacity (mAh/g)
sns.histplot(
    data=batches_list_df_unique,
    x="Initial specific discharge capacity (mAh/g)",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("Initial Specific Discharge Capacity (mAh/g)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
#
sns.histplot(
    data=batches_list_df_unique,
    x="First formation specific discharge capacity (mAh/g)",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("First formation specific discharge capacity (mAh/g)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Create the histogram with Initial efficiency (%) // First formation efficiency (%)
sns.histplot(
    data=batches_list_df_unique,
    x="Initial efficiency (%)",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("Initial efficiency (%)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()
#
sns.histplot(
    data=batches_list_df_unique,
    x="First formation efficiency (%)",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("First formation efficiency (%)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()

# Create the histogram with Cycles to 70% capacity
sns.histplot(
    data=batches_list_df_unique,
    x="Cycles to 70% capacity",
    hue="Group",           # Grouping by 'Group' for different colors
    palette=palette,       # Specify colors for each group
    multiple="stack",      # Stack the histograms for visual comparison
    kde=True,             # Optional: Add a kernel density estimate (set to True for smooth curves)
    bins=30,               # Number of bins (you can adjust this depending on your data)
)
plt.xlabel("Cycles to 70% capacity", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


#%%

# Create a combined DataFrame with Cycle, Specific discharge capacity, and Cell info for hover
all_data = []
for key, value in cells.items():
    value['Cell'] = key  # Add the cell identifier to the dataframe
    all_data.append(value)

# Combine the data from all cells into one DataFrame
df_combined = pd.concat(all_data)

# Define groups based on the Cell column
df_combined["Group"] = pd.cut(
    df_combined["Cell"], bins=[1, 17, 36], labels=["2-17", "18-36"], right=True
)

# Create a Plotly scatter plot
fig = px.scatter(df_combined,
                 x="Cycle",
                 y="Specific discharge capacity (mAh/g)",
                 color="Group",  # Use 'Group' to assign colors based on the groups
                 labels={"Cycle": "Cycle",
                         "Specific discharge capacity (mAh/g)": "Specific discharge capacity (mAh/g)"},
                 hover_data=["Cell"],  # Show the 'Cell' key on hover
                 color_discrete_map={"2-17": "blue", "18-36": "orange"})  # Define custom colors for groups

# Update layout to move the legend outside the plot
fig.update_layout(
    legend_title="Group",
    legend=dict(
        x=1.05,  # Place legend outside
        y=1,     # Position it at the top
        traceorder="normal",
        orientation="h",  # Horizontal legend
        font=dict(size=10)
    ),
    xaxis=dict(range=[4, 350])  # Set x-axis limits
)

# Show the plot
fig.show()
