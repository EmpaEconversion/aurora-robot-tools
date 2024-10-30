""" Lina Scholz

Script to read in images from folder, transform warped rectangle in straight rectangle,
detect centers of all parts, determine alignment vs. pressing tool (reference).
"""

import h5py
import math
import os
import json
import re
import numpy as np
import pandas as pd
import cv2

#%% CLASS WITH FUNCTIONS

class Alignment:
    def __init__(self, path):
        # TRANSFORMATION ---------------------------------------------------------------------------
        self.path = path # path to images
        self.savepath = path + "/processed" # path to save information
        self.transformed_images = [] # list to store transformed image arrays
        self.data_list = [] # list to store image data
        self.reference = [] # list to store images from pressing tool (step 0)
        self.ref = [] # list to store reference coordinates
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Parameter which might need to be changes if camera position changes ----------------------
        self.r = (210, 228) # (min, max)
        self.r_ellipse = (205, 240) # (min, max)

        # ALIGNMENT --------------------------------------------------------------------------------
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

        # Parameter which might need to be changed if camera position changes ----------------------
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
        # ------------------------------------------------------------------------------------------

    @staticmethod
    def _parse_filename(filename: str) -> list[dict]:
        """Take photo filename and returns dict of lists of press cell and step.

        Args:
            filename (str): the filename of the photo used

        Returns:
            list of dictionaries containing keys 'p', 'c', 's' for press, cell, step in the photo

        """
        pattern = re.compile(r"p(\d+)c(\d+)s(\d+)")
        matches = pattern.findall(filename)
        return [{"p": int(p), "c": int(c), "s": int(s)} for p, c, s in matches]

    # read files (images) from folder --------------------------------------------------------------
    def load_files(self) -> list:
        """ Loads images and stores them in list with filename and image array

        All images are loaded and stored in list with a tuple of their filename and 8 bit image
        array. Images from step 0 are stored additionally in a separate list to use them as a
        reference for the coordinate transformation.

        Returns:
            list: list containing filename and image array
        """
        print("load files")
        for filename in os.listdir(self.path):
            if filename.endswith('.h5'): # read all .h5 files
                filepath = os.path.join(self.path, filename)
                with h5py.File(filepath, 'r') as f:
                    content = f['image'][:]
                    content = content/np.max(content)*255 # convert to 8 bit
                    content = content.astype(np.uint8)
                    try:
                        if int(filename.split("_")[0].split("s")[1]) == 0: # images from step 0 as reference
                            self.reference.append((filename, content))
                        self.data_list.append((filename, content))
                    except (IndexError, ValueError) as e:
                        print(f"Error processing filename '{filename}': {e} \n only one cell or wrong filename")
                        # if there is no _ in the name (name is either wrong or only one cell)
                        if int(filename.split(".")[0].split("s")[1]) == 0: # images from step 0 as reference
                            self.reference.append((filename, content))
        return self.data_list

    # get reference coordinates --------------------------------------------------------------------
    def get_reference(self, circle_detection=False, ellipse_detection=True): # decide it ellipse or circle detection
        """ Takes each image from step 0 and gets the four corner coordinates of the pressing tools

        Args:
            circle_detection (bool): True if circles should be detected to find reference coordinates
            ellipse_detection (bool): True if ellipses should be detected to find reference coordinates (better)
        """
        print("get reference coordinates")
        for name, img in self.reference:
            img = cv2.convertScaleAbs(img, alpha=2, beta=0) # increase contrast

            if circle_detection:
                # Apply a Gaussian blur to the image before detecting circles (to improve detection)
                img = cv2.GaussianBlur(img, (9, 9), 2)
                # Apply Hough transform
                detected_circles = cv2.HoughCircles(img,
                                cv2.HOUGH_GRADIENT,
                                dp = 1,
                                minDist = 100,
                                param1 = 30, param2 = 50,
                                minRadius = self.r[0], maxRadius = self.r[1]) 
                # Extract center points and their pressing tool position
                coords = [] # list to store reference coordinates
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    for circle in detected_circles[0, :]:
                        coords.append((circle[0], circle[1]))
                self.ref.append((name, coords))

                # Draw all detected circles and save image to check quality of detection
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    cv2.circle(img, (a, b), r, (0, 0, 255), 10) # Draw the circumference of the circle
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 10) # Show center point drawing a small circle
                resized_img = cv2.resize(img, (1200, 800)) # Set image size
                # if folder doesn't exist, create it
                if not os.path.exists(self.savepath + "/reference_detected_circles"):
                    os.makedirs(self.savepath + "/reference_detected_circles")
                # Save the image with detected circles
                cv2.imwrite(self.savepath + f"/reference_detected_circles/{name.split(".")[0]}.jpg", resized_img)

            if ellipse_detection:
                coords = [] # list to store reference coordinates
                edges = cv2.Canny(img, 50, 150) # Edge detection for ellipses
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Find contours

                # Draw ellipses for each contour, constrained by aspect ratio and radius
                for contour in contours:
                    if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                        ellipse = cv2.fitEllipse(contour)
                        major_axis_length = ellipse[1][0]
                        minor_axis_length = ellipse[1][1]
                        # Calculate aspect ratio
                        if minor_axis_length > 0:  # Avoid division by zero
                            aspect_ratio = major_axis_length / minor_axis_length
                            # Calculate the average radius of the ellipse
                            avg_radius = (major_axis_length + minor_axis_length) / 4  # Approximate radius
                            # Constrain to shapes that are slightly non-circular and within the radius range
                            if 0.9 < aspect_ratio < 1.1 and self.r_ellipse[0] <= avg_radius <= self.r_ellipse[1]:
                                coords.append((ellipse[0], avg_radius))
                                cv2.ellipse(img, ellipse, (0, 255, 0), 10)  # Green color for ellipses
                                # Draw the center point
                                center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Convert coordinates to integers
                                cv2.circle(img, center, 5, (0, 255, 0), -1)

                # Filter out similar ellipses
                filtered_ellipses = []
                coords_ellipses = []
                for ellipse in coords:
                    (cx, cy), r = ellipse
                    # Check if the current ellipse is similar to any ellipses in the filtered list
                    if not any(np.sqrt((cx - fcx)**2 + (cy - fcy)**2) < 5 and abs(r - fr) < 5
                        for (fcx, fcy), fr in filtered_ellipses):
                            filtered_ellipses.append(ellipse)
                            center = (cx, cy)
                            coords_ellipses.append(center)
                self.ref.append((name, coords_ellipses))

                # Draw all detected ellipses and save image to check quality of detection
                resized_img = cv2.resize(img, (1200, 800)) # Set image size
                # if folder doesn't exist, create it
                if not os.path.exists(self.savepath + "/reference_detected_ellipses"):
                    os.makedirs(self.savepath + "/reference_detected_ellipses")
                # Save the image with detected ellipses
                cv2.imwrite(self.savepath + f"/reference_detected_ellipses/{name.split(".")[0]}.jpg", resized_img)

    # transform warped rectangle to straight rectangle ---------------------------------------------
    def transform_pixel_coordinates(self, save = False) -> list:
        """ Transform each image to get pressing tools in rectangular shape

        Transforms each image by the reference coordinates of the pressing tools to get pressing
        tool positions in rectangular shape instead of distorted warped shape.

        Args:
            save (bool): if transformed images should be saved as .h5 and .jpg files

        Returns:
            self.transformed_images (list): list with transformed image arrays
        """
        print("transform warped image in pixel coordinates")
        batch = 0 # iterate over each set of 1-6 batteries per pressing tool
        for name, img in self.data_list:
            height, width = img.shape[:2] # determine size of image
            ctr = self.ref[batch][1]
            ctr_sorted = [(0, 0), (0, 0), (0, 0), (0, 0)]
            for t in ctr:
                if (t[0] < 650) & (t[1] < 850): # edge of pressing tool position 1
                    ctr_sorted[0] = t
                elif (t[0] > 4750) & (t[1] < 850): # pressing tool position 3
                    ctr_sorted[1] = t
                elif (t[0] > 4750) & (t[1] > 2800): # pressing tool position 6
                    ctr_sorted[2] = t
                elif (t[0] < 650) & (t[1] > 2800): # pressing tool position 4
                    ctr_sorted[3] = t
                else:
                    pass
            ctr_sorted = np.float32(ctr_sorted)
            # Calculate new coordinates to correct for distortion:
            # fix coordinates of pressing tool position 1 and correct all other edges
            a1 = (ctr_sorted[0][0] + ((ctr_sorted[0][1] - ctr_sorted[1][1]))) / \
                (math.sin(math.atan(((ctr_sorted[0][1] - ctr_sorted[1][1])) / (ctr_sorted[1][0] - ctr_sorted[0][0]))))
            a2 = (ctr_sorted[0][1] + ((ctr_sorted[0][0] - ctr_sorted[3][0]))) / \
                (math.sin(math.atan(((ctr_sorted[0][0] - ctr_sorted[3][0])) / (ctr_sorted[3][1] - ctr_sorted[0][1]))))
            # corrected coordinates: a1 (corrected x coordinate), a2 (corrected y coordinate)
            pts2 = np.float32([ctr_sorted[0], [a1, ctr_sorted[0][1]], [a1, a2], [ctr_sorted[0][0], a2]])
            # Transform Perspective
            M = cv2.getPerspectiveTransform(ctr_sorted, pts2)
            transformed_image = cv2.warpPerspective(img, M, (width, height))
            self.transformed_images.append((name, transformed_image)) # store image

            if save:
                # Save the transformed_image to jpg
                if not os.path.exists(f'{self.savepath}/transformed_jpg'):
                    os.makedirs(f'{self.savepath}/transformed_jpg')
                cv2.imwrite(f'{self.savepath}/transformed_jpg/{name.split(".")[0]}.jpg', transformed_image)
                # Save the transformed_image to an .h5 file
                if not os.path.exists(f'{self.savepath}/transformed_h5'):
                    os.makedirs(f'{self.savepath}/transformed_h5')
                p = self.savepath + f"/transformed_h5/{name.split(".")[0]}.h5"
                with h5py.File(p, 'w') as h5_file:
                    h5_file.create_dataset('image', data=transformed_image)

            # update batch number in case batch is finished to change reference for transformation
            try:
                if int(name.split("_")[0].split("s")[1]) == 9: # check if last step (9) is reached
                    batch += 1
            except (IndexError, ValueError) as e:
                print(f"Error processing name '{name}': {e} \n only one cell in pressing tools")
                if int(name.split(".")[0].split("s")[1]) == 9: # in case there is only one cell in this batch
                    batch += 1
        return self.transformed_images

    # get circle coordinates -----------------------------------------------------------------------
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

        for name, img in self.transformed_images: # iterate over all images
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

    # correct for z distortion from thickness ------------------------------------------------------
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

    # get alignment numbers ------------------------------------------------------------------------
    def alignment_number(self, z_corrected = False) -> pd.DataFrame:
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

    # convert pixel to mm --------------------------------------------------------------------------
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
if __name__ == '__main__':
    # PARAMETER
    # path to files
    path = 'G:/Limit/Lina Scholz/robot_files_gen14'
    # path = "G:/Limit/Lina Scholz/robot_files/kigr_gen5"

    # EXECUTE
    obj = Alignment(path)
    files = obj.load_files() # load all images from folder
    reference = obj.get_reference() # get center points from first step of assembly as reference
    transformed = obj.transform_pixel_coordinates() # transform pixel coordinates to get "rectangular shape"
    images_detected = obj.get_coordinates() # get coordinates of all circles
    # images_z_corrected = obj.z_correction() # correct coordinates for z distortion due to thickness
    images_alignment = obj.alignment_number() # get alignment
    images_alignment_mm = obj.pixel_to_mm() # get alignment number in mm


    # TODO: ----------------------------------------------------------------------------------------
    # combine these objects avoiding the intermediate save of the .h5 files
    # include alignment class
    # images_detected = obj.get_coordinates() # get coordinates of all circles
    # images_alignment = obj.alignment_number() # get alignment
    # images_alignment_mm = obj.pixel_to_mm() # get alignment number in mm

    # somehow save this data
    # probably we add a table in the chemspeed db with all the data
    # then we can process that in output_csv and deal with that later

    # save as (json string) e.g.
    # data = {'cell': 1, 'data': [{'step': 1, 'x_mm': 0.5, 'y_mm': 0.12},{'step': 2, 'x_mm': 0.24, 'y_mm': 0.06}]}
    # json_str = json.dumps(data)
