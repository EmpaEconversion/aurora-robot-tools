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

#%% CLASS

class ProcessImages:
    def __init__(self, path):
        # TRANSFORMATION ---------------------------------------------------------------------------
        self.path = path # path to images
        self.ref = [] # list with references (coords and corresponding cell numbers)
        self.data_list = [] # list to store image data
        self.df = pd.DataFrame(columns=["cell", "step", "press", "array"]) # data frame for all data

        # Parameter which might need to be changes if camera position changes ----------------------
        self.press_position = [[0, 0], [95, 0], [190, 0], [190, 100], [95, 100], [0, 100]]
        self.mm_coords = np.float32([[0, 0], [190, 0], [190, 100], [0, 100]])
        self.mm_to_pixel = 10
        self.offset_mm = 20 # mm
        self.r = (210, 228) # (min, max) radius of pressing tool for reference detection
        self.r_ellipse = (205, 240) # (min, max) radius of pressing tool for reference detection

        # ALIGNMENT --------------------------------------------------------------------------------
        self.alignment_df = pd.DataFrame()
        # Parameter which might need to be changes if camera position changes ----------------------
        # radius of all parts from cell in mm (key corresponds to step)
        self.r_part = {0: (9.5, 10.5), 1: (9.5, 10.5), 2: (6.75, 8), 3: (7, 8), 4: (7.5, 8.5),
                       5: (7.7, 8.5), 6: (6.25, 7.5), 7: (7, 8.25), 8: (6.25, 7.7), 9: (9.5, 10.5),
                       10: (7, 11)}
        # parameter for HoughCircles (param1, param2)
        self.params =[(30, 50), (30, 50), (5, 10), (30, 50), (30, 50),
                      (30, 50), (5, 25), (30, 50), (5, 20), (30, 50), (30, 50)]

    def _parse_filename(self, filename: str) -> list[dict]:
        """Take photo filename and returns dict of lists of press cell and step.

        Args:
            filename (str): the filename of the photo used

        Returns:
            list of dictionaries containing keys 'p', 'c', 's' for press, cell, step in the photo
        """
        pattern = re.compile(r"p(\d+)c(\d+)s(\d+)")
        matches = pattern.findall(filename)
        # [{"p": 1, "c": 1, "s": 0}, {"p": 3, "c": 2, "s": 0}]
        return [{"p": int(p), "c": int(c), "s": int(s)} for p, c, s in matches]

    def _get_references(self, filenameinfo: list[dict], img: np.array, ellipse_detection=True) -> tuple[np.array, list]:
        """ Takes each image from step 0 and gets the four corner coordinates of the pressing tools

        Args:
            filenameinfo (list[dicts]): list of dicts with press, cell, step
            img (array): image array
            ellipse_detection (bool): True if circles should be detected to find reference coordinates

        Returns:
            tuple with transformation matrix and list of cell numbers
        """
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # increase contrast
        ref_image_name = "_".join(str(d["c"]) for d in filenameinfo) # name with all cells belonging to reference

        if ellipse_detection:
            coordinates, _, image_with_circles = self._detect_ellipses(img, self.r_ellipse)
        else:
            coordinates, _, image_with_circles = self._detect_circles(img, self.r)

        # Draw all detected ellipses and save image to check quality of detection
        height, width = image_with_circles.shape[:2] # Get height, width
        resized_img = cv2.resize(image_with_circles, (width, height)) # Set image size
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/reference"):
            os.makedirs(self.path + "/reference")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/reference/{ref_image_name}.jpg", resized_img)

        transformation_M = self._get_transformation_matrix(coordinates)
        return (transformation_M, [d["c"] for d in filenameinfo]) # transformation matrix with cell numbers

    def _detect_ellipses(self, img: np.array, r: tuple) -> tuple[list[list], np.array]:
        """ Takes image, detects ellipses of pressing tools and provides list of coordinates.

        Args:
            img (array): image array

        Return:
            coords_ellipses (list[list]): list with all six center coordinates of pressing tools
        """
        # center, rad, image_with_circles = self._detect_ellipses(img, r)
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
                    if 0.9 < aspect_ratio < 1.1 and r[0] <= avg_radius <= r[1]:
                        coords.append((ellipse[0], avg_radius))
                        cv2.ellipse(img, ellipse, (0, 255, 0), 10)  # Green color for ellipses
                        # Draw the center point
                        center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Convert coordinates to integers
                        cv2.circle(img, center, 5, (0, 255, 0), -1)
        # Filter out similar ellipses
        filtered_ellipses = []
        coords_ellipses = []
        r_ellipses = []
        for ellipse in coords:
            (cx, cy), r = ellipse
            # Check if the current ellipse is similar to any ellipses in the filtered list
            if not any(np.sqrt((cx - fcx)**2 + (cy - fcy)**2) < 10 and abs(r - fr) < 10
                for (fcx, fcy), fr in filtered_ellipses):
                    filtered_ellipses.append(ellipse)
                    coords_ellipses.append((cx, cy))
                    r_ellipses.append(r)
        return coords_ellipses, r_ellipses, img

    def _detect_circles(self, img: np.array, radius: tuple, params: tuple) -> tuple[list[list], list[list], np.array]:
        """ Takes image, detects circles of pressing tools and provides list of coordinates.

        Args:
            img (array): image array
            radius (tuple): (minimum_radius, maximum_radius) to detect
            params (tuple): (param1, param2) for HoughCircles

        Return:
            coords_circles (list[list]): list with all center coordinates of pressing tools
        """
        # Apply Hough transform
        detected_circles = cv2.HoughCircles(img,
                        cv2.HOUGH_GRADIENT,
                        dp = 1,
                        minDist = 500,
                        param1 = params[0], param2 = params[1],
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

    def _get_transformation_matrix(self, centers: list[tuple]) -> np.array:
        """ Takes center points of reference image and gets transformation matrix.

        Args:
            centers (list[tuple]): list with the four corner coordinates

        Returns:
            M (array): transformation matrix
        """
        pts2 = np.float32((self.mm_coords + self.offset_mm)*self.mm_to_pixel)
        # Sort center coordinates in correct order for transformation matrix
        y_values = [center[1] for center in centers] # Extract the y-values
        mean_y = sum(y_values) / len(y_values) # Calculate the mean of the y-values
        # Split the list based on the median y-value
        lower_y_group = [center for center in centers if center[1] < (mean_y - 500)]
        higher_y_group = [center for center in centers if center[1] > (mean_y + 500)]
        # Sort top and bottom points by x
        top_half_sorted = sorted(lower_y_group, key=lambda x: x[0])
        bottom_half_sorted = sorted(higher_y_group, key=lambda x: x[0])
        # Arrange in desired order: upper left, upper right, lower right, lower left
        centers_sorted = np.float32([top_half_sorted[0], top_half_sorted[-1],
                                     bottom_half_sorted[-1], bottom_half_sorted[0]])
        # Transform Perspective
        M = cv2.getPerspectiveTransform(centers_sorted, pts2) # transformation matrix
        return M

    def _transform_split(self, img: np.array, m: np.array, filename: str) -> list[np.array]:
        """ Takes image and transformation matrix and returns transformed image.

        Args:
            img (array): image array
            m (array): transformation matrix
            filename (str): filename

        Returns:
            cropped_images (array): transformed image splitted into subsections
        """
        transformed_image = cv2.warpPerspective(img, m,
                                                ((190+ 2* self.offset_mm)*self.mm_to_pixel,
                                                 (100+ 2* self.offset_mm)*self.mm_to_pixel))
        # for cross check save image: # TODO delete later?
        height, width = transformed_image.shape[:2] # Get height, width
        resized_img = cv2.resize(transformed_image, (width, height)) # Set image size
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/transformed"):
            os.makedirs(self.path + "/transformed")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/transformed/{filename}.jpg", resized_img)
        # Crop the image
        cropped_images = {}
        for i, c in enumerate(self.press_position):
            # set zero in case it gives a negative number
            bottom_right_y = max((c[1] + 2*self.offset_mm) * self.mm_to_pixel, 0)
            bottom_right_x = max((c[0] + 2*self.offset_mm) * self.mm_to_pixel, 0)
            top_left_y = bottom_right_y - 400
            top_left_x = bottom_right_x - 400
            cropped_image = transformed_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            # for cross check save image:
            height, width = cropped_image.shape[:2] # Get height, width
            resized_img = cv2.resize(cropped_image, (width, height)) # Set image size
            # if folder doesn't exist, create it
            if not os.path.exists(self.path + "/image_sections"):
                os.makedirs(self.path + "/image_sections")
            # Save the image with detected ellipses
            cv2.imwrite(self.path + f"/image_sections/{filename}_pos_{i+1}.jpg", resized_img)
            cropped_images[i] = cropped_image
        return cropped_images

    def _get_alignment(self, step_a: int, step_b: int) -> dict[tuple]:
        """ Get missalignment of the two steps.

        Args:
            step_1 (int): first part for missalignment calculation
            step_2 (int): second part for missalignemnt calculation

        Return:
            missalignment_dict (dict): missalignemnt per cell in (x, y, z)
        """
        missalignment_dict_x = {}
        missalignment_dict_y = {}
        missalignment_dict_z = {}
        cells = self.df["cell"].unique()
        for i, cell in enumerate(cells):
            cell_df = self.df[self.df["cell"] == cell]
            part_a_x = cell_df.loc[cell_df['step'] == step_a, 'x'].values
            part_a_y = cell_df.loc[cell_df['step'] == step_a, 'y'].values
            part_b_x = cell_df.loc[cell_df['step'] == step_b, 'x'].values
            part_b_y = cell_df.loc[cell_df['step'] == step_b, 'y'].values

            x = (float(part_a_x[0]) - float(part_b_x[0])) / self.mm_to_pixel
            y = (float(part_a_y[0]) - float(part_b_y[0])) / self.mm_to_pixel
            z = round(math.sqrt(x**2 + y**2), 3)
            missalignment_dict_x[cell] = x
            missalignment_dict_y[cell] = y
            missalignment_dict_z[cell] = z
        return missalignment_dict_x, missalignment_dict_y, missalignment_dict_z

    def _preprocess_image(self, image: np.array, step: int) -> np.array:
        """ Takes image and applies preprocessing steps (blur, contrast)

        Args:
            image (array): image array
            step (int): robot assembly step

        Return:
            processed_image (array): processed image
        """
        if step == 2:
            # Apply a Gaussian blur to reduce noise and improve detection accuracy
            processed_image = cv2.convertScaleAbs(image, alpha=3, beta=0) # increase contrast
            processed_image = cv2.GaussianBlur(processed_image, (9, 9), 2)
        else: # no preprossessing
            processed_image = image
        return processed_image

    def load_files(self) -> list[tuple]:
        """ Loads images and stores them in list with filename and image array

        All images are loaded and stored in list with a tuple of their filename and 8 bit image
        array. Images from step 0 are stored additionally in a separate list to use them as a
        reference for the coordinate transformation.

        Returns:
            list: list containing filename, information from image name and image array
        """
        for filename in os.listdir(self.path):
            if filename.endswith('.h5'): # read all .h5 files
                filepath = os.path.join(self.path, filename)
                with h5py.File(filepath, 'r') as f:
                    content = f['image'][:]
                    content = content/np.max(content)*255 # convert to 8 bit
                    content = content.astype(np.uint8) # image array
                info = self._parse_filename(filename) # extract info from filename
                if all(d["s"] == 0 for d in info): # if step 0, get reference coordinates
                    matrix = self._get_references(info, content) # transformation matrix with cell numbers
                    self.ref.append(matrix)
                self.data_list.append((filename, info, content)) # store info and image array
        return

    def store_data(self) -> pd.DataFrame:
        """ For each image array transform image and store image sections in DataFrame.

        Returns:
            self.df (DataFrame): columns cell, step, press, transformed image section, center coordinates
        """
        for name, information, image in self.data_list:
            for array, numbers in self.ref:
                if numbers == [d["c"] for d in information]: # find matching transformation matrix for cell numbers
                    transformation_matrix = array
            image_sections = self._transform_split(image, transformation_matrix, name)
            for dictionary in information:
                position = int(dictionary["p"])-1
                row = [dictionary["c"], dictionary["s"], dictionary["p"], image_sections[position]]
                self.df.loc[len(self.df)] = row
        return

    def get_centers(self) -> pd.DataFrame:
        """ Detects centers of parts for each image section in data frame

        Returns:
            self.df (data frame): data frame with column of center coordinates added
        """
        x = [] # list to store coordinates
        y = []
        radius = [] # list to store radius
        for index, row in self.df.iterrows():
            r = tuple(int(x * self.mm_to_pixel) for x in self.r_part[row["step"]])
            img = self._preprocess_image(row["array"], row["step"])
            parameter = self.params[row["step"]]
            if row["step"] == "type in step of part which should be detected as ellipse":
                center, rad, image_with_circles = self._detect_ellipses(img, r, parameter)
            else: # detect circle
                center, rad, image_with_circles = self._detect_circles(img, r, parameter)
            # Assuming center is expected to be a list containing a tuple
            if center is not None and isinstance(center, list) and len(center) > 0:
                x.append(center[0][0])
                y.append(center[0][1])
                radius.append(rad[0]/self.mm_to_pixel)
            else:
                # Handle the case where center is None or not as expected
                x.append(np.nan)
                y.append(np.nan)
                radius.append(None)
            # for cross check save image:
            height, width = image_with_circles.shape[:2] # Get height, width
            resized_img = cv2.resize(image_with_circles, (width, height)) # Set image size
            # if folder doesn't exist, create it
            if not os.path.exists(self.path + "/detected_circles"):
                os.makedirs(self.path + "/detected_circles")
            # Save the image with detected ellipses
            filename = f"c{row["cell"]}_p{row["press"]}_s{row["step"]}"
            cv2.imwrite(self.path + f"/detected_circles/{filename}.jpg", resized_img)
        self.df["x"] = x
        self.df["y"] = y
        self.df["r_mm"] = radius
        return

    def get_alignment(self) -> pd.DataFrame:
        """ Get alignemnt between all specified parts into data frame.

        Return:
            self.alignment_df (data frame): alignments of all cells for different parts.
        """
        self.alignment_df = self.df.groupby('cell')['press'].first().reset_index()
        anode_cathode_x, anode_cathode_y, anode_cathode_z = self._get_alignment(2, 6)
        spring_press_x, spring_press_y, spring_press_z = self._get_alignment(8, 0)
        spacer_press_x, spacer_press_y, spacer_press_z = self._get_alignment(7, 0)
        self.alignment_df["anode/cathode_z"] = self.alignment_df['cell'].map(anode_cathode_z)
        self.alignment_df["spring/press_z"] = self.alignment_df['cell'].map(spring_press_z)
        self.alignment_df["spacer/press_z"] = self.alignment_df['cell'].map(spacer_press_z)
        return

    def save(self) -> pd.DataFrame:
        """ Saves data frames with all coordinates, redius and alignment.
        """
        if not os.path.exists(self.path + "/data"):
            os.makedirs(self.path + "/data")
        with pd.ExcelWriter(self.path + "/data/data.xlsx") as writer:
            self.df.to_excel(writer, sheet_name='coordinates', index=False)
            self.alignment_df.to_excel(writer, sheet_name='alignment', index=False)
        return self.df, self.alignment_df

#%% RUN CODE
if __name__ == '__main__':

    # PARAMETER
    folderpath = "C:/lisc_gen14"

    obj = ProcessImages(folderpath)
    obj.load_files()
    obj.store_data()
    obj.get_centers()
    obj.get_alignment()
    coordinates_df, alignment_df = obj.save()

    print(coordinates_df)
    print(alignment_df)

    # base_string = "241004_kigr_gen5_"
    # Create a list with formatted strings
    # sample_ID = [f"{base_string}{str(i).zfill(2)}" for i in range(1, 37)]

    base_string = "241022_lisc_gen14_"
    sample_ID = [
    '241022_lisc_gen14_2-13_02', '241022_lisc_gen14_2-13_03', '241022_lisc_gen14_2-13_04', '241022_lisc_gen14_2-13_05',
    '241022_lisc_gen14_2-13_06', '241022_lisc_gen14_2-13_07', '241022_lisc_gen14_2-13_08', '241022_lisc_gen14_2-13_09',
    '241022_lisc_gen14_2-13_10', '241022_lisc_gen14_2-13_11', '241022_lisc_gen14_2-13_12', '241022_lisc_gen14_2-13_13',
    '241022_lisc_gen14_14_36_14', '241022_lisc_gen14_14_36_15', '241022_lisc_gen14_14_36_16', '241022_lisc_gen14_14_36_17',
    '241022_lisc_gen14_14_36_18', '241022_lisc_gen14_14_36_19', '241022_lisc_gen14_14_36_20', '241022_lisc_gen14_14_36_21',
    '241022_lisc_gen14_14_36_22', '241022_lisc_gen14_14_36_23', '241022_lisc_gen14_14_36_24', '241022_lisc_gen14_14_36_25',
    '241022_lisc_gen14_14_36_26', '241022_lisc_gen14_14_36_27', '241022_lisc_gen14_14_36_28', '241022_lisc_gen14_14_36_29',
    '241022_lisc_gen14_14_36_30', '241022_lisc_gen14_14_36_31', '241022_lisc_gen14_14_36_32', '241022_lisc_gen14_14_36_33',
    '241022_lisc_gen14_14_36_34', '241022_lisc_gen14_14_36_35', '241022_lisc_gen14_14_36_36'
    ]

    alignments = alignment_df["anode/cathode_z"]
    alignments["sample_ID"] = sample_ID
    alignments.to_csv(f"{folderpath}/data/{base_string}alignment.csv", index=False)
