""" Lina Scholz

Script to read in images from folder and detect circles and their alignment.
"""

import cv2
import os
import math
import statistics
import numpy as np
import pandas as pd
import h5py

#%% CLASS WITH FUNCTIONS

class ALIGNMENT:
    def __init__(self, path):
        self.path = path
        self.savepath = path + "/processed" # path to save information
        self.data_list = [] # list to store images (name and array)
        self.df_images = pd.DataFrame()
        self.columns = ["pos", "cell",
                        "s0_coords", "s1_coords", "s2_coords", "s3_coords", "s4_coords", "s5_coords",
                        "s6_coords", "s7_coords", "s8_coords", "s9_coords", "s10_coords", # pixel coordinates
                        "s0_r", "s1_r", "s2_r", "s3_r", "s4_r", "s5_r",
                        "s6_r", "s7_r", "s8_r", "s9_r", "s10_r"] # pixel radius

        """"s1_align", "s2_align", "s4_align", "s5_align", "s6_align", "s7_align",
            "s8_align", "s9_align", "s10_align", # alignment vs. pressing tool in pixel
            "s1_align_mm", "s2_align_mm", "s4_align_mm", "s5_align_mm", "s6_align_mm", "s7_align_mm",
            "s8_align_mm", "s9_align_mm", "s10_align_mm"] # alignment vs. pressing tool in mm"""

        # Parameter which might need to be changed if camera position changes
        """Steps:
        0:pressing tool, 1:bottom part, 2:anode, 4:separator, 5:electrolyte,
        6:cathode, 7:spacer, 8:spring, 9:top part, 10:after pressing"""
        self.r_min = {0: 200, 1: 200, 2: 145, 4: 160, 5: 170, 6: 140, 7: 150, 8: 140, 9: 140, 10: 160} # min circle r
        self.r_max = {0: 230, 1: 230, 2: 175, 4: 183, 5: 190, 6: 162, 7: 178, 8: 170, 9: 185, 10: 190}  # max circle r

        """Rectangles constraining the place of the circle for each pressing tool:"""
        self.pos_1 = [(350, 650), (600, 800)] # top left, bottom right corner of rectangle
        self.pos_2 = [(2600, 650), (2760, 800)]
        self.pos_3 = [(4600, 650), (5100, 800)]
        self.pos_4 = [(350, 2800), (650, 3050)]
        self.pos_5 = [(2600, 2800), (2760, 3050)]
        self.pos_6 = [(4600, 2800), (5100, 3050)]
        self.rectangles = [self.pos_1, self.pos_2, self.pos_3, self.pos_4, self.pos_5, self.pos_6]

        """Coordinates for correction of z-distortion due to thickness of parts (from thickness_distortion.py)"""
        self.z_corr_1 = [(0.9, 1.8), (0.0, 1.8), (0.75, 1.8), (0.9, 0.9), (0.0, 0.9), (0.75, 0.9)] # after step 1
        self.z_corr_4 = [(4.65, 9.3), (0.0, 9.3), (3.875, 9.3), (4.65, 4.65), (0.0, 4.65), (3.875, 4.65)] # after step 4
        self.z_corr_7 = [(7.65, 15.3), (0.0, 15.3), (6.375, 15.3),
                         (7.65, 7.65), (0.0, 7.65), (6.375, 7.65)] # after step 7

    # read files in list ----------------------------------------------
    def read_files(self):
        print("\nStep: read files in list")
        for filename in os.listdir(self.path):
            if filename.endswith('.h5'):
                filepath = os.path.join(self.path, filename)
                with h5py.File(filepath, 'r') as f:
                    content = f['image'][:]
                    # convert to 8 bit
                    content = content/np.max(content)*255
                    content = content.astype(np.uint8)
                    self.data_list.append((filename, content)) # store filename and image array
        return self.data_list

    # get circle coordinates ----------------------------------------------
    def get_coordinates(self) -> pd.DataFrame:
        """Gets coordinates and radius of all parts and stores them in a data frame

        For each image in the list of images, the step as well as the positions and cell numbers
        are extracted and the image processes accordingly (contrast & gaussian blur) for better
        circle detection. Hough transform is applied to detect circles with the defined radius for
        this part. The position of the circle is constraint by the rectangles defined above. The
        extracted coordinates are assigned to each cell and stored in the data frame. The circles as
        well as the constraining rectangle are drawn in the image and saved as jpg.

        Return:
            self.df_images (data frame): storing position, cell number, coordinate and radius (pixel)
        """
        print("\nStep: detect circles and get circle coordinates")
        positions = [] # list to store positions
        cell_numbers = [] # list to store cell numbers
        coordinates = {0: [], 1: [], 2: [], 4: [], 5:[], 6: [], 7: [], 8: [], 9: [], 10:[]} # coordinates of all steps
        radius = {0: [], 1: [], 2: [], 4: [], 5:[], 6: [], 7: [], 8: [], 9: [], 10:[]} # stores radius of all steps

        for name, img in self.data_list: # iterate over all images
            try:
                step = int(name.split("_")[0].split("s")[1]) # get step
            except (IndexError, ValueError) as e:
                print(f" Error processing name '{name}': {e}\n only one cell or wring filename")
                step = int(name.split(".")[0].split("s")[1]) # in case there is only cell
            if step == 0: # assign position and cell number in data frame
                current_positions = []
                img = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # increase contrast
                img = cv2.GaussianBlur(img, (5, 5), 2) # Gaussian blur to image before detecting (to improve detection)
                string = name.split(".")[0] # get name as string
                for i in range(len(string.split("_"))):
                    if len(string.split("_")) > 1:
                        current_positions.append(str(string.split("_")[i].split("c")[0][-2:]))
                        positions.append(str(string.split("_")[i].split("c")[0][-2:]))
                        cell_numbers.append(str(string.split("_")[i].split("c")[1].split("s")[0]))
                    else: # only one cell
                        print(" only one cell in pressing tools")
                        current_positions.append(str(string.split("c")[0][-2:]))
                        positions.append(str(string.split("c")[0][-2:]))
                        cell_numbers.append(str(string.split("c")[1].split("s")[0]))
            elif step == 2: # increase constrast for the anode
                img = cv2.convertScaleAbs(img, alpha=2, beta=0)
                img = cv2.GaussianBlur(img, (5, 5), 2) # Gaussian blur to image before detecting
            elif step == 6: # no contrast change for cathode
                img = cv2.GaussianBlur(img, (5, 5), 2) # Gaussian blur to image before detecting
            elif step == 8:
                img = cv2.convertScaleAbs(img, alpha=1.25, beta=0) # increase contrast
            else: # default values to process image
                img = cv2.convertScaleAbs(img, alpha=1.25, beta=0) # increase contrast
                img = cv2.GaussianBlur(img, (5, 5), 2) # Gaussian blur to image before detecting

            # Apply Hough transform
            detected_circles = cv2.HoughCircles(img,
                            cv2.HOUGH_GRADIENT,
                            dp = 1,
                            minDist = 100,
                            param1 = 30, param2 = 50,
                            minRadius = self.r_min[step], maxRadius = self.r_max[step])

            # Extract center points and their pressing tool position
            # dictionaries updated with new coordinates & radius for each image (intermediate save before adding to df)
            coords_buffer_dict = {}
            r_buffer_dict = {}
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))
                for circle in detected_circles[0, :]:
                    # assign circle pressing tool
                    # constrain by rectangles to avoid too many circles
                    if (circle[1] > self.pos_4[0][1]) & (circle[1] < self.pos_4[1][1]): # position 4, 5, 6
                        if (circle[0] > self.pos_4[0][0]) & (circle[0] < self.pos_4[1][0]): # position 4
                            if "04" in current_positions:
                                coords_buffer_dict[4] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[4] = circle[2] # radius
                        elif (circle[0] > self.pos_5[0][0]) & (circle[0] < self.pos_5[1][0]): # position 5
                            if "05" in current_positions:
                                coords_buffer_dict[5] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[5] = circle[2] # radius
                        elif (circle[0] > self.pos_6[0][0]) & (circle[0] < self.pos_6[1][0]): # position 6
                            if "06" in current_positions:
                                coords_buffer_dict[6] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[6] = circle[2] # radius
                        else:
                            print(f"\n circle in lower row couldnt be assigned: ({circle[0]}, {circle[1]})")
                            # Create a mask that identifies incorrectly positioned circles to be remove
                            # axis=-1 to compare along the last dimension
                            mask = ~np.all(np.isin(detected_circles, circle), axis=-1)
                            detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape

                    elif (circle[1] > self.pos_1[0][1]) & (circle[1] < self.pos_1[1][1]): # position 1, 2, 3
                        if (circle[0] > self.pos_1[0][0]) & (circle[0] < self.pos_1[1][0]): # position 1
                            if "01" in current_positions:
                                coords_buffer_dict[1] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[1] = circle[2] # radius
                        elif (circle[0] > self.pos_2[0][0]) & (circle[0] < self.pos_2[1][0]): # position 2
                            if "02" in current_positions:
                                coords_buffer_dict[2] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[2] = circle[2] # radius
                        elif (circle[0] > self.pos_3[0][0]) & (circle[0] < self.pos_3[1][0]): # position 3
                            if "03" in current_positions:
                                coords_buffer_dict[3] = [circle[0], circle[1]]  # (x, y) coordinates
                                r_buffer_dict[3] = circle[2] # radius
                        else:
                            print(f"\n circle in upper row couldnt be assigned: ({circle[0]}, {circle[1]})")
                            # Create a mask that identifies incorrectly positioned circles to be remove
                            mask = ~np.all(np.isin(detected_circles, circle), axis=-1)
                            detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape
                    else:
                        print(f"\n circle couldnt be assigned for any pressing tool: ({circle[0]}, {circle[1]})")
                        # Create a mask that identifies incorrectly positioned circles to be remove
                        mask = ~np.all(np.isin(detected_circles, circle), axis=-1)
                        detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape

                # Draw all detected circles and save image to check quality of detection
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    cv2.circle(img, (a, b), r, (0, 0, 255), 10) # Draw the circumference of the circle
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 10) # Show center point drawing a small circle
                for rect in self.rectangles: # Draw constraining rectagles for pressing tool position area
                    cv2.rectangle(img, rect[0], rect[1], (0, 0, 255), 10) # Red rectangle with 10-pixel thickness
                resized_img = cv2.resize(img, (1200, 800))
                # if folder doesn't exist, create it
                if not os.path.exists(self.savepath + "/detected_circles"):
                    os.makedirs(self.savepath + "/detected_circles")
                    # Save the image with detected circles
                cv2.imwrite(self.savepath + f"/detected_circles/{name.split(".")[0]}.jpg", resized_img)

                # add circles which were not detected with zeros
                current_positions_int = [int(pos) for pos in current_positions] # Convert entries to integers
                for pos in current_positions_int: # Check each position in `current_positions_int`
                    if pos not in coords_buffer_dict:
                        coords_buffer_dict[pos] = [0, 0]
                    if pos not in r_buffer_dict:
                        r_buffer_dict[pos] = 0
                # Create a new dictionary with keys sorted by the specified order, skipping missing keys
                key_order = [1, 3, 5, 2, 4, 6] # order of pressing tool positions in string name of image file
                coords_buffer_dict = {key: coords_buffer_dict[key] for key in key_order if key in coords_buffer_dict}
                r_buffer_dict = {key: r_buffer_dict[key] for key in key_order if key in r_buffer_dict}
                # Create a list of all values without the keys
                coords_buffer_list = list(coords_buffer_dict.values())
                r_buffer_list = list(r_buffer_dict.values())
                # add values to list to collect values
                c = coordinates[step]
                c.extend(coords_buffer_list)
                coordinates[step] = c
                r = radius[step]
                r.extend(r_buffer_list)
                radius[step] = r
            else: # check if there are no cells in the pressing tools, or if just no part could be detected
                print(" detected circles is none")
                if len(current_positions) != 0:
                    coords_buffer_list = [[0, 0] for _ in range(len(current_positions))] # add zeros per default
                    r_buffer_list = [0 for _ in range(len(current_positions))]
                    c = coordinates[step]
                    c.extend(coords_buffer_list)
                    coordinates[step] = c
                    r = radius[step]
                    r.extend(r_buffer_list)
                    radius[step] = r

        # fill values into dataframe
        for num, column in enumerate(self.columns):
            if column == "pos":
                self.df_images[column] = positions
            elif column == "cell":
                self.df_images[column] = cell_numbers
            elif num < 13:
                key = num - 2
                if key != 3:
                    self.df_images[column] = coordinates[key]
            elif (num < 24) & (num > 12):
                key = num - 13
                if key != 3:
                    self.df_images[column] = radius[key]
        # get cell number and position as integers not string
        self.df_images['pos'] = self.df_images['pos'].astype(int)
        self.df_images['cell'] = self.df_images['cell'].astype(int)
        return self.df_images

    # correct for z distortion from thickness ----------------------------
    def z_correction(self) -> pd.DataFrame:
        """Correct the pixel coordinates by accounting for the thickness of the thickest parts.

        The amount to correct for is determined in the script "thickness_distortion.py" by using the
        height of the pressing tool and the according shift in the centers from the bottom to the
        top.

        Return:
            self.df_images (data frame): storing position, cell number, (un)corrected coordinate and radius (pixel)
        """
        print("\nStep: correct for z distortion of thickness of parts")
        for index, row in self.df_images.iterrows():
            # correct distortion for step 1, 4, 7 (bottom, separator, spacer)
            pos = row["pos"]
            # correct for z distortion
            s1_corr = self.z_corr_1[(pos - 1)]
            s4_corr = self.z_corr_4[(pos - 1)]
            s7_corr = self.z_corr_7[(pos - 1)]
            for i, col_name in enumerate(self.columns[3:13]):
                if col_name != "s3_coords": # exclude step 3
                    step_num = col_name.split("_")[0].replace('s', '') # step number
                    new_column_name = f"s{step_num}_coords_corr" # column name to store corrected coordinates
                    if i == 0: # no thickness to correct for
                        x = int(row[col_name][0])
                        y = int(row[col_name][1])
                    elif i < 4: # correct for thickness of bottom part
                        x = int(row[col_name][0]) - s1_corr[0]
                        y = int(row[col_name][1]) + s1_corr[1] # add since y is increasing from top down
                    elif i < 7: # additionally correct for thickness of separator
                        x = int(row[col_name][0]) - s4_corr[0]
                        y = int(row[col_name][1]) + s4_corr[1]
                    elif i >= 7: # additionally correct for thickness of spacer
                        x = int(row[col_name][0]) - s7_corr[0]
                        y = int(row[col_name][1]) + s7_corr[1]
                    self.df_images._set_value(index, new_column_name, [x, y])

        return self.df_images

    # get alignment numbers ----------------------------------------------
    def alignment_number(self, z_corrected = True) -> pd.DataFrame:
        """Determine alignment (x, y, z) vs. pressing tool for each part

        Args:
            z_corrected (bool): whether to account for thickness of parts

        Return:
            self.df_images (data frame): storing position, cell number, (un)corrected coordinate,
                                         alignment numbers and radius (pixel)
        """
        print("\nStep: determine alignment in pixel")
        # Define the column names to be added
        alignment_columns = ["s1_align", "s2_align", "s4_align", "s5_align",
                             "s6_align", "s7_align", "s8_align", "s9_align", "s10_align"]
        for col in alignment_columns: # Add each new column with None or pd.NA as empty values
            self.df_images[col] = pd.NA
        if z_corrected: # account for the thickness of the parts by taking corrected coordinates
            # list of column names for coordinates which are corrected for z distortion (thickness)
            corr_col_names = [f"s{i}_coords_corr" for i in range(11) if i != 3]
            for index, row in self.df_images.iterrows():
                x_ref = row["s0_coords_corr"][0]
                y_ref = row["s0_coords_corr"][1]
                for col_name in corr_col_names:
                    step_num = col_name.split("_")[0].replace('s', '') # step number
                    n = f"s{step_num}_align" # column name of where to store alignment entry
                    x = int(x_ref) - int(row[col_name][0]) # KeyError: 's3_coords'
                    y = int(y_ref) - int(row[col_name][1])
                    z = math.sqrt(x**2 + y**2)
                    self.df_images._set_value(index, str(n), (x, y, z))
        else: # do not account for thickness of parts
            for index, row in self.df_images.iterrows():
                x_ref = row["s0_coords"][0]
                y_ref = row["s0_coords"][1]
                for col_name in self.columns[3:13]:
                    if col_name != "s3_coords": # exclude step 3
                        step_num = col_name.split("_")[0].replace('s', '') # step number
                        n = f"s{step_num}_align" # column name of where to store alignment entry
                        x = int(x_ref) - int(row[col_name][0]) # KeyError: 's3_coords'
                        y = int(y_ref) - int(row[col_name][1])
                        z = math.sqrt(x**2 + y**2)
                        self.df_images._set_value(index, str(n), (x, y, z))
        return self.df_images

    # convert pixel to mm ----------------------------------------------
    def pixel_to_mm(self, with_radius = True): # decide whether to convert by radius or rectangle coordinates
        """Convert pixel to mm

        Args:
            with_radius (bool): either convert with the radius or with the size of the rectangle
                                between the pressing tools

        Return:
            self.df_images (data frame): storing position, cell number, (un)corrected coordinate,
                                         alignment numbers and radius (pixel & mm)
        """
        print("\nStep: convert pixel values to mm")
        if with_radius:
            # pixel value of radius
            pixel = (sum(self.df_images["s0_r"].to_list())/len(self.df_images["s0_r"].to_list()) * 2)
            mm = 20 # radius in mm
            pixel_to_mm = mm/pixel # convert
            print( "pixel to mm: " + str(pixel_to_mm) + " mm/pixel")
        else:
            # get positions of lower corners of rectangle between pressing tools
            pos_4_coords = self.df_images[self.df_images["pos"] == 4]["s0_coords"].tolist()
            pos_6_coords = self.df_images[self.df_images["pos"] == 6]["s0_coords"].tolist()
            # pixel value of distance between lower pressing tools
            pixel = statistics.median([abs(item4[0] - item6[0]) for item4, item6 in zip(pos_4_coords, pos_6_coords)])
            mm = 190 # distance in mm
            pixel_to_mm = mm/pixel # convert
            print(" pixel to mm: " + str(pixel_to_mm) + " mm/pixel")
        # missalignment to mm
        for i in list(range(1, 11)):
            if i != 3:
                # add alignment in mm
                self.df_images[f"s{i}_align_mm"] = [(round(x * pixel_to_mm, 3), round(y * pixel_to_mm, 3), 
                                                     round(z * pixel_to_mm, 3)) for x, y, z in self.df_images[f"s{i}_align"].to_list()]
        # radius to mm
        for i in list(range(0, 11)):
            if i != 3:
                # add radius in mm
                self.df_images[f"s{i}_r_mm"] = [round(r * pixel_to_mm, 3) for r in self.df_images[f"s{i}_r"].to_list()]

        # Save data
        self.df_images.sort_values(by="cell", inplace=True)
        if not os.path.exists(self.savepath + "/data"):
            os.makedirs(self.savepath + "/data")
        self.df_images.to_excel(self.savepath + "/data/data.xlsx")

        return self.df_images

#%% RUN CODE

# PARAMETER
path = "G:/Limit/Lina Scholz/robot_files_gen14"

# EXECUTE
obj = ALIGNMENT(path)
imgages = obj.read_files() # list with all images given as a list
images_detected = obj.get_coordinates() # get coordinates of all circles
images_z_corrected = obj.z_correction() # correct coordinates for z distortion due to thickness
images_alignment = obj.alignment_number() # get alignment
images_alignment_mm = obj.pixel_to_mm() # get alignment number in mm

print(images_alignment_mm.head())

#%% SAVE

# images_alignment_mm.sort_values(by="cell", inplace=True)
# if not os.path.exists(path + "/data"):
#     os.makedirs(path + "/data")
# images_alignment_mm.to_excel(path + "/data/data.xlsx")


