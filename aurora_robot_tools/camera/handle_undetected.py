""" Lina Scholz

Script to handle undetected circles.
"""

import math
import os
import json
import numpy as np
import pandas as pd
import cv2

def _detect_circles(img: np.array, radius: tuple) -> tuple[list[list], list[list], np.array]:
        """ Takes image, detects circles of pressing tools and provides list of coordinates.

        Args:
            img (array): image array
            radius (tuple): (minimum_radius, maximum_radius) to detect

        Return:
            coords_circles (list[list]): list with all center coordinates of pressing tools
        """
        # Apply Hough transform
        detected_circles = cv2.HoughCircles(img,
                        cv2.HOUGH_GRADIENT,
                        dp = 1,
                        minDist = 100,
                        param1 = 30, param2 = 50,
                        minRadius = radius[0], maxRadius = radius[1])
        # Extract center points and their pressing tool position
        coords_circles = [] # list to store coordinates
        r_circles = [] # list to store radius
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                coords_circles.append((circle[0], circle[1]))
                r_circles.append(circle[2])
            # Draw all detected circles and save image to check quality of detection
            detected_circles = np.uint16(np.around(detected_circles))
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                cv2.circle(img, (a, b), r, (0, 0, 255), 5) # Draw the circumference of the circle
                cv2.circle(img, (a, b), 1, (0, 0, 255), 5) # Show center point drawing a small circle
        else:
            coords_circles = None
            r_circles = None
        return coords_circles, r_circles, img

def _get_alignment(data: pd.DataFrame, step_a: int, step_b: int) -> dict[tuple]:
    """ Get missalignment of the two steps.

    Args:
        data (data frame): all coordinates and radius
        step_1 (int): first part for missalignment calculation
        step_2 (int): second part for missalignemnt calculation

    Return:
        missalignment_dict (dict): missalignemnt per cell in (x, y, z)
    """
    missalignment_dict_x = {}
    missalignment_dict_y = {}
    missalignment_dict_z = {}
    cells = data["cell"].unique()
    for i, cell in enumerate(cells):
        cell_df = data[data["cell"] == cell]
        part_a_x = cell_df.loc[cell_df['step'] == step_a, 'x'].values
        part_a_y = cell_df.loc[cell_df['step'] == step_a, 'y'].values
        part_b_x = cell_df.loc[cell_df['step'] == step_b, 'x'].values
        part_b_y = cell_df.loc[cell_df['step'] == step_b, 'y'].values

        x = (float(part_a_x[0]) - float(part_b_x[0])) / mm_to_pixel
        y = (float(part_a_y[0]) - float(part_b_y[0])) / mm_to_pixel
        z = round(math.sqrt(x**2 + y**2), 3)
        missalignment_dict_x[cell] = x
        missalignment_dict_y[cell] = y
        missalignment_dict_z[cell] = z
    return missalignment_dict_x, missalignment_dict_y, missalignment_dict_z

#%% Get missing center coordinates

path = "C:/lisc_gen14"
steps = [2, 6]
mm_to_pixel = 10
r_part = {0: (9.5, 10.5), 1: (9.5, 10.5), 2: (6.5, 8), 3: (7, 8), 4: (7.5, 8.5),
          5: (7.7, 8.5), 6: (6.5, 7.5), 7: (7, 8.25), 8: (6, 7.75), 9: (9.5, 10.5), 10: (7, 11)}

df = pd.read_excel(f"{path}/data/data.xlsx", sheet_name = "coordinates")
df_alignment = pd.read_excel(f"{path}/data/data.xlsx", sheet_name = "alignment")
for s in steps:
    folder_path = f"{path}/not_detected/{s}"
    # List to store filename and content as a tuple
    files_content = []
    centers = {}

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Only process files if they are Excel or CSV
        if filename.endswith(".jpg"):
            img = cv2.imread(f"{folder_path}/{filename}")
            img_array = np.array(img)
            # convert to 8 bit
            img = img/np.max(img)*255
            img = img.astype(np.uint8)
            # Append a tuple with filename and content to the list
            files_content.append((filename, img))
            cell = int(filename.split(".")[0].split("_")[0].split("c")[1])

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            r = tuple(int(x * mm_to_pixel) for x in (r_part[2][0], r_part[2][1]))
            center, rad, image_detected = _detect_circles(img, r)
            if center is not None and rad is not None:
                centers[(cell, s)] = center[0], rad[0]/mm_to_pixel
            else:
                centers[(cell, s)] = (0, 0), 0
    print(centers)

    for key, value in centers.items():
        row_index = df[(df["cell"] == key[0]) & (df["step"] == key[1])].index[0]
        df.iloc[row_index, df.columns.get_loc("x")] = value[0][0]
        df.iloc[row_index, df.columns.get_loc("y")] = value[0][1]
        df.iloc[row_index, df.columns.get_loc("r_mm")] = round(value[1], 3)

df.to_excel(f"{path}/data/data_corrected.xlsx", sheet_name = "coordinates")
with pd.ExcelWriter(f"{path}/data/data_corrected.xlsx") as writer:
    df.to_excel(writer, sheet_name='coordinates', index=False)
    df_alignment.to_excel(writer, sheet_name='alignment', index=False)

#%% Get all alignment numbers

data = pd.read_excel(f"{path}/data/data_corrected.xlsx", sheet_name = "coordinates")

data_alignment = data.groupby('cell')['press'].first().reset_index()
anode_cathode_x, anode_cathode_y, anode_cathode_z = _get_alignment(data, 2, 6)
spring_press_x, spring_press_y, spring_press_z = _get_alignment(data, 8, 0)
spacer_press_x, spacer_press_y, spacer_press_z = _get_alignment(data, 7, 0)
data_alignment["anode/cathode"] = data_alignment['cell'].map(anode_cathode_z)
data_alignment["spring/press"] = data_alignment['cell'].map(spring_press_z)
data_alignment["spacer/press"] = data_alignment['cell'].map(spacer_press_z)

with pd.ExcelWriter(f"{path}/data/data_corrected.xlsx") as writer:
    data.to_excel(writer, sheet_name='coordinates', index=False)
    data_alignment.to_excel(writer, sheet_name='alignment', index=False)

#%% Save to plot

alignment = pd.DataFrame()

sample_ID = [
 '241022_lisc_gen14_2_13_2', '241022_lisc_gen14_2_13_3', '241022_lisc_gen14_2_13_4', '241022_lisc_gen14_2_13_5',
 '241022_lisc_gen14_2_13_6', '241022_lisc_gen14_2_13_7', '241022_lisc_gen14_2_13_8', '241022_lisc_gen14_2_13_9',
 '241022_lisc_gen14_2_13_10', '241022_lisc_gen14_2_13_11', '241022_lisc_gen14_2_13_12', '241022_lisc_gen14_2_13_13',
 '241022_lisc_gen13_14_36_14', '241022_lisc_gen13_14_36_15', '241022_lisc_gen13_14_36_16', '241022_lisc_gen13_14_36_17',
 '241022_lisc_gen13_14_36_18', '241022_lisc_gen13_14_36_19', '241022_lisc_gen13_14_36_20', '241022_lisc_gen13_14_36_21',
 '241022_lisc_gen13_14_36_22', '241022_lisc_gen13_14_36_23', '241022_lisc_gen13_14_36_24', '241022_lisc_gen13_14_36_25',
 '241022_lisc_gen13_14_36_26', '241022_lisc_gen13_14_36_27', '241022_lisc_gen13_14_36_28', '241022_lisc_gen13_14_36_29',
 '241022_lisc_gen13_14_36_30', '241022_lisc_gen13_14_36_31', '241022_lisc_gen13_14_36_32', '241022_lisc_gen13_14_36_33',
 '241022_lisc_gen13_14_36_34', '241022_lisc_gen13_14_36_35', '241022_lisc_gen13_14_36_36'
]

anode_cathode_z.pop(1)

alignment["sample_ID"] = sample_ID
alignment["alignment"] = list(anode_cathode_z.values())

alignment.to_csv(f"{path}/data/alignment_data.csv", index=False)


