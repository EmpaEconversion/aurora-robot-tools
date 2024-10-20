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
fig.savefig(plot_path + "/" + unique_filename, format='png')

# --------------------------

# TODO: create overlapping plot





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
fig.savefig(plot_path + "/" + unique_filename, format='png')