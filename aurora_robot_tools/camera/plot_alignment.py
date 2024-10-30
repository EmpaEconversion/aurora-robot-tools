""" Lina Scholz

Script to plot the alignment
"""

import os
import ast
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%%
save = True

# Linas Cells (lisc_gen14)
path = "G:/Limit/Lina Scholz/robot_files_gen14/processed"
plot_path = "G:/Limit/Lina Scholz/robot_files_gen14/processed/plots"
df_images = pd.read_excel(f"{path}/data/data.xlsx")
df_images.sort_values(by="cell", inplace=True)

# Grahams Cells (kigr_gen5)
# path = "G:/Limit/Lina Scholz/robot_files_names/transformed"
# plot_path = "G:/Limit/Lina Scholz/robot_files_names/transformed/plots_new"
# df_images = pd.read_excel(f"{path}/data/data.xlsx")
# df_images.sort_values(by="cell", inplace=True)

#%% ANODE VS CATHODE

# anode: s2_align_mm & cathode: s6_align_mm
fig, ax = plt.subplots(layout="tight", figsize=(16, 10))

# anode
anode = [ast.literal_eval(item) for item in df_images["s2_align_mm"].to_list()] # Convert each string to a tuple
anode_x = [x for x, y, z in anode]
anode_y = [y for x, y, z in anode]
# cathode
cathode = [ast.literal_eval(item) for item in df_images["s6_align_mm"].to_list()] # Convert each string to a tuple
cathode_x = [x for x, y, z in cathode]
cathode_y = [y for x, y, z in cathode]
# x-value
x_values = [x1 - x2 for x1, x2 in zip(anode_x, cathode_x)]
# y-value
y_values = [y1 - y2 for y1, y2 in zip(anode_y, cathode_y)]
alignment = [math.sqrt(x**2 + y**2) for x, y, in zip(x_values, y_values)]
df_images["alignment"] = alignment

# plot alignment in mm
colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#313695']
# Grouping the data by the pressing tool position column and plotting each group separately
for i, (group, df_group) in enumerate(df_images.groupby('pos')):
    ax.scatter(df_group['cell'].tolist(), df_group["alignment"],
               color=colors[i], s=50, label=f'Position {group}')
# labeling
ax.set_xlabel("cell number", fontsize=18)
ax.set_ylabel("alignment offset [mm]", fontsize=18)
ax.set_xticks(df_images["cell"], minor=True)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([-0.1, 3])
ax.set_title("Anode vs. Cathode Alignment", fontsize=18)
ax.grid(True, axis='y')
ax.legend(title="Pressing Tool", fontsize=14)
# Display the plot
plt.show()

if save:
    unique_filename = "anode_vs_cathode.png"
    i = 0
    while os.path.exists(plot_path + "/" + unique_filename):
        unique_filename = f"{unique_filename.split(".")[0].split("-")[0]}-{i}.png"
        i += 1
    # create path if not existing
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # save plot
    fig.savefig(plot_path + "/" + unique_filename, format='png')
df_images.drop(columns=["alignment"], inplace=True)

# --------------------------
# Grid size
grid_size = 6

# List of misalignments (y-values correspond to batches, x-values to pressing tool position)
missalignments = [(x/10, y/10) for x, y in zip(x_values, y_values)] # in um
cat_missalign = [missalignments[0:6], missalignments[6:12], missalignments[12:18],
                 missalignments[18:24], missalignments[24:30], missalignments[30:36]]

cat_radius = df_images["s6_r_mm"].to_list() # cathode radius
cat_radius = [x/10 for x in cat_radius]
radii_cat = [cat_radius[0:6], cat_radius[6:12], cat_radius[12:18],
             cat_radius[18:24], cat_radius[24:30], cat_radius[30:36]]

ano_radius = df_images["s4_r_mm"].to_list() # anode radius
ano_radius = [x/10 for x in ano_radius]
radii_ano = [ano_radius[0:6], ano_radius[6:12], ano_radius[12:18],
             ano_radius[18:24], ano_radius[24:30], ano_radius[30:36]]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))

# Loop over grid points and draw circles
for i in range(grid_size):
    for j in range(grid_size):
        try:
            # Get the base radius for each point
            base_radius = radii_ano[j][i]
            # First circle (centered at (i+1, j+1) because we want labels 1 to 6)
            circle1 = plt.Circle((3*(i+1), 2*(j+1)), base_radius, color='blue', fill=False)
            # Get the misalignment and radius for the second circle
            misalign_x, misalign_y = cat_missalign[j][i]
            misalign_radius = radii_cat[j][i]
            # Second circle (misaligned by (misalign_x, misalign_y))
            # minus missalign_y to reverse axis (from image to plot)
            circle2 = plt.Circle((3*(i+1) + misalign_x, 2*(j+1) - misalign_y), misalign_radius, color='red', fill=False)
            # Add circles to plot
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            # Plot points for the centers of both circles (smaller markers)
            ax.plot(3 * (i + 1), 2 * (j + 1), 'bo', markersize=3)  # Anode circle center
            ax.plot(3 * (i + 1) + misalign_x, 2 * (j + 1) + misalign_y, 'ro', markersize=3)  # Cathode circle center
        except (IndexError, TypeError, ValueError) as e:
            print(f" Error plotting circles at batch {i}, pos {j}: {e}\n")
            print(f" No more circles in list to plot: batch {i}, pos {j}")

# Set limits and aspect ratio
ax.set_xlim(0, 3 * grid_size + 3)
ax.set_ylim(0, 2 * grid_size + 2)
ax.set_aspect('equal')
# Invert the y-axis so that it increases from top to bottom
# ax.invert_yaxis()
# Set axis labels from 1 to 6 at the correct positions
ax.set_xticks([3, 6, 9, 12, 15, 18])
ax.set_xticklabels([1, 3, 5, 2, 4, 6])
ax.set_yticks([2, 4, 6, 8, 10, 12])
# ax.set_yticklabels(range(1, grid_size + 1)[::-1])  # Reverse the labels for the y-axis
ax.set_yticklabels([1, 2, 3, 4, 5, 6])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# axis labels
ax.set_xlabel("Pressing Tool Position", fontsize=16)
ax.set_ylabel("Production Batch", fontsize=16)
# Create legend
base_patch = mpatches.Patch(color='blue', label='Anode')
second_circle_patch = mpatches.Patch(color='red', label='Cathode')
ax.legend(handles=[base_patch, second_circle_patch], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.show()

if save:
    unique_filename = "anode_vs_cathode_circles.png"
    i = 0
    while os.path.exists(plot_path + "/" + unique_filename):
        unique_filename = f"{unique_filename.split(".")[0].split("-")[0]}-{i}.png"
        i += 1
    # create path if not existing
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # save plot
    fig.savefig(plot_path + "/" + unique_filename, format='png')

#%% XXX INPUT VS PRESSING TOOL

input = 7
name = "spring"
string = f"s{input}_align_mm"

fig, ax = plt.subplots(layout="tight", figsize=(16, 10))

part = [ast.literal_eval(item) for item in df_images[string].to_list()] # Convert each string to a tuple
alignment = [z for x, y, z in part]
df_images["alignment"] = alignment

# plot alignment in mm
colors = ['#d73027', '#fc8d59', '#fee08b', '#91bfdb', '#4575b4', '#313695']
# Grouping the data by the pressing tool position column and plotting each group separately
for i, (group, df_group) in enumerate(df_images.groupby('pos')):
    ax.scatter(df_group['cell'].tolist(), df_group["alignment"],
               color=colors[i], s=50, label=f'Position {group}')
# labeling
ax.set_xlabel("cell number", fontsize=18) 
ax.set_ylabel("alignment offset [mm]", fontsize=18)
ax.set_xticks(df_images["cell"], minor=True)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([-0.1, 3])
ax.set_title(f"{name} vs. Pressing Tool Alignment", fontsize=18)
ax.grid(True, axis='y')
ax.legend(title="Pressing Tool", fontsize=14)
plt.show()

if save:
    unique_filename = f"{name}_vs_origin.png"
    i = 0
    while os.path.exists(plot_path + "/" + unique_filename):
        unique_filename = f"{unique_filename.split(".")[0].split("-")[0]}-{i}.png"
        i += 1
    # create path if not existing
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # save plot
    fig.savefig(plot_path + "/" + unique_filename, format='png')
df_images.drop(columns=["alignment"], inplace=True)

# --------------------------
# Grid size
grid_size = 6

# List of misalignments (y-values correspond to batches, x-values to pressing tool position)
x_missalign = [x for x, y, z in part]
y_missalign = [y for x, y, z in part]
missalignments = [(x/10, y/10) for x, y in zip(x_missalign, y_missalign)] # in um
cat_missalign = [missalignments[0:6], missalignments[6:12], missalignments[12:18],
                 missalignments[18:24], missalignments[24:30], missalignments[30:36]]

string_r = f"s{input}_r_mm"
part_radius = df_images[string_r].to_list() # input radius
part_radius = [x/10 for x in part_radius]
radii_part = [part_radius[0:6], part_radius[6:12], part_radius[12:18],
              part_radius[18:24], part_radius[24:30], part_radius[30:36]]

tool_radius = df_images["s0_r_mm"].to_list() # pressing tool radius
tool_radius = [x/10 for x in tool_radius]
radii_tool = [tool_radius[0:6], tool_radius[6:12], tool_radius[12:18],
              tool_radius[18:24], tool_radius[24:30], tool_radius[30:36]]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(12, 10))

# Loop over grid points and draw circles
for i in range(grid_size):
    for j in range(grid_size):
        try:
            # Get the base radius for each point
            base_radius = radii_tool[j][i]
            # First circle (centered at (i+1, j+1) because we want labels 1 to 6)
            circle1 = plt.Circle((4*(i+1), 3*(j+1)), base_radius, color='blue', fill=False)
            # Get the misalignment and radius for the second circle
            misalign_x, misalign_y = cat_missalign[j][i]
            misalign_radius = radii_part[j][i]
            # Second circle (misaligned by (misalign_x, misalign_y))
            # minus missalign_y to reverse axis (from image to plot)
            circle2 = plt.Circle((4*(i+1) + misalign_x, 3*(j+1) - misalign_y), misalign_radius, color='red', fill=False)
            # Add circles to plot
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            # Plot points for the centers of both circles (smaller markers)
            ax.plot(4 * (i + 1), 3 * (j + 1), 'bo', markersize=3)  # Anode circle center
            ax.plot(4 * (i + 1) + misalign_x, 3 * (j + 1) + misalign_y, 'ro', markersize=3)  # Cathode circle center
        except (IndexError, TypeError, ValueError) as e:
            print(f" Error plotting circles at batch {i}, pos {j}: {e}\n")
            print(f" No more circles in list to plot: batch {i}, pos {j}")

# Set limits and aspect ratio
ax.set_xlim(0, 4 * grid_size + 4)
ax.set_ylim(0, 3 * grid_size + 3)
ax.set_aspect('equal')
# Set axis labels from 1 to 6 at the correct positions
ax.set_xticks([4, 8, 12, 16, 20, 24])
ax.set_xticklabels([1, 3, 5, 2, 4, 6])
ax.set_yticks([3, 6, 9, 12, 15, 18])
ax.set_yticklabels([1, 2, 3, 4, 5, 6])
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# axis labels
ax.set_xlabel("Pressing Tool Position", fontsize=16)
ax.set_ylabel("Production Batch", fontsize=16)
# Create legend
base_patch = mpatches.Patch(color='blue', label='Pressing Tool')
second_circle_patch = mpatches.Patch(color='red', label=name)
ax.legend(handles=[base_patch, second_circle_patch], bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)
plt.show()

if save:
    unique_filename = f"{name}_vs_origin_circles.png"
    i = 0
    while os.path.exists(plot_path + "/" + unique_filename):
        unique_filename = f"{unique_filename.split(".")[0].split("-")[0]}-{i}.png"
        i += 1
    # create path if not existing
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    # save plot
    fig.savefig(plot_path + "/" + unique_filename, format='png')
