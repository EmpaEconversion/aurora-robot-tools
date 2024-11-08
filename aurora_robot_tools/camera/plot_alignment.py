""" Lina Scholz

Script to plot the alignment
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
save = False

folderpath_alignment = "G:/Limit/Lina Scholz/Alignment"
folderpath_cellperformance = "G:/Limit/Lina Scholz/Data"

alignment = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment.csv")
alignment.set_index('sample_ID', inplace=True)
spring = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment_spring.csv")
spacer = pd.read_csv(f"{folderpath_alignment}/241022_lisc_gen14_alignment_spacer.csv")
alignment["spring"] = spring["spring/press"].to_list()
alignment["spacer"] = spacer["spacer/press"].to_list()
print("\n")
print(alignment)

all_keys = set()
x_columns = ["anode/cathode", "spring", "spacer"]
plot_strings = ["Cycles to 90% capacity", "Cycles to 80% capacity", "Initial efficiency (%)"]
# "Formation C", "Efficiency (%)"

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

# show json data
print("\n")
print(all_keys)
print("\n")
print(alignment)


#%% ANODE VS CATHODE

# Create a figure and axis array for a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

# Loop through each x_column and each y_column in plot_strings
for i, x_col in enumerate(x_columns):
    for j, y_col in enumerate(plot_strings):
        axes[i, j].scatter(alignment[x_col], alignment[y_col], marker='o')
        axes[i, j].set_title(f"{y_col} vs {x_col} alignment", fontsize=9)
        axes[i, j].set_xlabel(f"{x_col} alignment [mm]", fontsize=8)
        axes[i, j].set_ylabel(y_col, fontsize=8)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
plt.show()
