""" Lina Scholz

Script to compare manually improved detection with only automated detection.

"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

#%%

folder = "C:/lisc_gen14/json"

# Load the adjusted JSON file
name_adj = "alignment_adjusted.lisc_gen14.json"
data_dir_adj = os.path.join(folder, name_adj)
with open(data_dir_adj, 'r') as file:
    data_adj = json.load(file)
df_adj = pd.DataFrame(data_adj["alignment"]) # Convert the "alignment" key into a DataFrame
print(df_adj.head())
# Save back as data frame
name_list = name_adj.split(".")
name_list.pop()
name_save = ".".join(map(str, name_list))
with pd.ExcelWriter(os.path.join("C:/lisc_gen14/data", f"{name_save}.xlsx")) as writer:
    df_adj.to_excel(writer, sheet_name='difference', index=False)

# Load the automated JSON file
data_dir = os.path.join(folder, "alignment.lisc_gen14.json")
with open(data_dir, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data["alignment"]) # Convert the "alignment" key into a DataFrame
print(df.head())

# Compute difference
numerical_columns = ["r_mm", "dx_mm_corr", "dy_mm_corr"]
if df.shape == df_adj.shape:
    # Compute the difference
    df_diff = df[numerical_columns] - df_adj[numerical_columns]
    df_diff = pd.concat([df_diff, df[["cell", "step", "press"]]], axis=1)
else:
    raise ValueError("DataFrames df and df_adj must have the same shape")
print(df_diff.head())

# Show differences in plot
columns_to_plot = ["dx_mm_corr", "dy_mm_corr"]
steps_to_plot = [0, 2, 6, 8]

# Loop through each step to create a figure
for step in steps_to_plot:
    # Filter the DataFrame for the current step
    step_data = df[df["step"] == step]

    # Create a new figure for this step
    fig, axes = plt.subplots(1, len(columns_to_plot), figsize=(15, 5), sharey=False)  # 1 row, 3 columns

    # Plot each column as a subplot
    for i, column in enumerate(columns_to_plot):
        axes[i].hist(step_data[column], bins=36, alpha=0.7, color='blue')
        axes[i].set_title(f"{column} (Step {step})")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True)

    # Add a title for the entire figure
    fig.suptitle(f"Distributions for Step {step}", fontsize=16)
    plt.tight_layout()
    plt.show()
