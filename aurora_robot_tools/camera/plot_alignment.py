"""
Script to read in images from folder and detect circles

"""

import cv2
import os
import ast
import math
import numpy as np
import pandas as pd
import h5py
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%%

path = "G:/Limit/Lina Scholz/robot_files_names/transformed"
plot_path = "G:/Limit/Lina Scholz/robot_files_names/transformed/plot_alignment"
df_images = pd.read_excel(f"{path}/data/data.xlsx")

#%% ANODE VS CATHODE

# anode: s2_align [mm] & cathode: s6_align [mm]
fig, ax = plt.subplots(layout="tight", figsize=(16, 10))

# anode
anode = [ast.literal_eval(item) for item in df_images["s2_align [mm]"].to_list()] # Convert each string to a tuple
anode_x = [x for x, y, z in anode]
anode_y = [y for x, y, z in anode]
# cathode
cathode = [ast.literal_eval(item) for item in df_images["s6_align [mm]"].to_list()] # Convert each string to a tuple
cathode_x = [x for x, y, z in cathode]
cathode_y = [y for x, y, z in cathode]
# x-value
x_values = [x1 - x2 for x1, x2 in zip(anode_x, cathode_x)]
# y-value
y_values = [y1 - y2 for y1, y2 in zip(anode_y, cathode_y)]
alignment = [math.sqrt(x**2 + y**2) for x, y, in zip(x_values, y_values)]

# plot alignment in mm
ax.scatter(df_images["cell"].tolist(), alignment, color="black", s=50)
# labeling
ax.set_xlabel("cell number", fontsize=20) 
ax.set_ylabel("alignment offset [mm]", fontsize=20)
ax.set_xticks(df_images["pos"], minor=True)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([-0.1, 5])
ax.set_title("Anode vs. Cathode Alignment")

unique_filename = "anode_vs_cathode.png"
i = 0
while os.path.exists(plot_path + "/" + unique_filename):
    unique_filename = f"{unique_filename}_{i}"
    i += 1
# create path if not existing
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# save plot
# fig.savefig(plot_path + "/" + unique_filename, format='png')

# --------------------------

# TODO: create overlapping plot
# Grid size
grid_size = 6

# List of misalignments (y-values correspond to batches, x-values to pressing tool position)
missalignments = [(x/100, y/100) for x, y in zip(x_values, y_values)] # in um
cat_missalign = [missalignments[0:6], missalignments[6:12], missalignments[12:18], missalignments[18:24], missalignments[24:30], missalignments[30:36]]

cat_radius = df_images["s6_r"].to_list() # cathode radius
cat_radius = [x / 1000 for x in cat_radius]
radii_cat = [cat_radius[0:6], cat_radius[6:12], cat_radius[12:18], cat_radius[18:24], cat_radius[24:30], cat_radius[30:36]]

ano_radius = df_images["s4_r"].to_list() # anode radius
ano_radius = [x / 1000 for x in ano_radius]
radii_ano = [ano_radius[0:6], ano_radius[6:12], ano_radius[12:18], ano_radius[18:24], ano_radius[24:30], ano_radius[30:36]]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(16, 12))

# Loop over grid points and draw circles
for i in range(grid_size):
    for j in range(grid_size):
        # Get the base radius for each point
        base_radius = radii_ano[j][i]
        
        # First circle (centered at (i+1, j+1) because we want labels 1 to 6)
        circle1 = plt.Circle((2* (i + 1), 1.5*(j + 1)), base_radius, color='blue', fill=False)
        
        # Get the misalignment and radius for the second circle
        misalign_x, misalign_y = cat_missalign[j][i]
        misalign_radius = radii_cat[j][i]
        
        # Second circle (misaligned by (misalign_x, misalign_y))
        circle2 = plt.Circle((2 * (i + 1) + misalign_x, 1.5 * (j + 1) + misalign_y), misalign_radius, color='red', fill=False)

        # Add circles to plot
        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # Plot points for the centers of both circles (smaller markers)
        ax.plot(2 * (i + 1), 1.5 * (j + 1), 'bo', markersize=3)  # Anode circle center
        ax.plot(2 * (i + 1) + misalign_x, 1.5 * (j + 1) + misalign_y, 'ro', markersize=3)  # Cathode circle center

# Set limits and aspect ratio
ax.set_xlim(0, 2 * grid_size + 1)
ax.set_ylim(0, 1.5 * grid_size + 1)
ax.set_aspect('equal')

# Set axis labels from 1 to 6 at the correct positions
ax.set_xticks([2, 4, 6, 8, 10, 12])
ax.set_xticklabels(range(1, grid_size + 1))
ax.set_yticks([1.5, 3, 4.5, 6, 7.5, 9])
ax.set_yticklabels(range(1, grid_size + 1))

# Create legend
base_patch = mpatches.Patch(color='blue', label='Base')
second_circle_patch = mpatches.Patch(color='red', label='Second Circle')
ax.legend(handles=[base_patch, second_circle_patch], loc='upper right')

# Display the plot
plt.show()




#%% SPRING VS PRESSING TOOL

# spring: s7_align [mm] 
fig, ax = plt.subplots(layout="tight", figsize=(16, 10))

spring = [ast.literal_eval(item) for item in df_images["s7_align [mm]"].to_list()] # Convert each string to a tuple
alignment = [z for x, y, z in spring]

# plot alignment in mm
ax.scatter(df_images["cell"].tolist(), alignment, color="black", s=50)
# labeling
ax.set_xlabel("cell number", fontsize=20) 
ax.set_ylabel("alignment offset [mm]", fontsize=20)
ax.set_xticks(df_images["pos"], minor=True)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim([-0.1, 5])
ax.set_title("Spring vs. Pressing Tool Alignment")

unique_filename = "spring_vs_origin.png"
i = 0
while os.path.exists(plot_path + "/" + unique_filename):
    unique_filename = f"{unique_filename}_{i}.png"
    i += 1
# create path if not existing
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# save plot
# fig.savefig(plot_path + "/" + unique_filename, format='png')