""" Lina Scholz

Script to plot the alignment
"""

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
save = False

folderpath_alignment = "G:/Limit/Lina Scholz/Alignment/lisc_gen14/data"
folderpath_cellperformance = "G:/Limit/Lina Scholz/Data"

# anode/cathode
alignment = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment.csv")
alignment.set_index('sample_ID', inplace=True)
alignment = alignment.rename(columns={'anode/cathode': 'alignment'})

# spring
spring = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment_spring.csv")
spring.set_index('sample_ID', inplace=True)
spring = spring.rename(columns={'spring/press': 'alignment'})

# spacer
spacer = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment_spacer.csv")
spacer.set_index('sample_ID', inplace=True)
spacer = spacer.rename(columns={'spacer/press': 'alignment'})

# get performance
all_keys = set()
plot_strings = ["Cycles to 80% capacity",
                "Cycles to 85% energy",
                "Last specific discharge capacity (mAh/g)"]

# possible strings:
"""
Capacity loss (%), Initial efficiency (%), First formation specific discharge capacity (mAh/g),
First formation efficiency (%), Cycles to 85% energy, Last specific discharge capacity (mAh/g),
Initial specific discharge capacity (mAh/g)
"""

for string in plot_strings:
    cell_data = {}
    # Loop through all files in the folder
    for filename in os.listdir(folderpath_cellperformance):
        if filename.endswith('.json'):  # Check if the file is a JSON file
            file_path = os.path.join(folderpath_cellperformance, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                json_data = json.load(file)

                # Extract the data associated with the key "data"
                if "data" in json_data and isinstance(json_data["data"], dict):
                    name = filename.split(".")[1]
                    cell_data[name] = (json_data["data"][string])
                    all_keys.update(json_data["data"].keys())

    # add cell performance data to aligment data frame
    alignment[string] = alignment.index.map(cell_data)
    spring[string] = spring.index.map(cell_data)
    spacer[string] = spacer.index.map(cell_data)

# Drop all rows with any NaN values
alignment = alignment.dropna()
spring = spring.dropna()
spacer = spacer.dropna()
# drop all rows with no reasonable alignment offset
alignment = alignment[alignment['alignment'] <= 5]
spring = spring[spring['alignment'] <= 5]
spacer = spacer[spacer['alignment'] <= 5]
# create list with all alignment data frames
dataframes = [alignment, spring, spacer]

# show json data
print("\n")
print(all_keys)
print("\n")
print(alignment)

#%% ANODE VS CATHODE

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle("Alignment vs. Cell Performance", fontsize=16)
parts = ["anode/cathode", "spring", "spacer"]

# Loop through each DataFrame and column
for row, df in enumerate(dataframes):
    for col, y_col in enumerate(plot_strings):
        # Scatter plot
        axes[row, col].scatter(df['alignment'], df[y_col], color='blue')

        # Calculate and plot regression line
        slope, intercept = np.polyfit(df['alignment'], df[y_col], 1)
        regression_line = slope * df['alignment'] + intercept
        axes[row, col].plot(df['alignment'], regression_line, color='red')

        # Set titles and labels
        axes[row, col].set_title(f"{y_col} vs {parts[row]}", fontsize=10)
        axes[row, col].set_xlabel(f"{parts[row]}", fontsize=8)
        axes[row, col].set_ylabel(y_col, fontsize=8)

# Adjust layout for readability
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
