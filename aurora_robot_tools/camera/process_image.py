""" Copyright © 2024, Empa, Lina Scholz, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

The script does the following steps:
    1. The images are read in from the folder with the run_ID of the cells, where they have been
    stored during the assembly with the coin-cell robot.
    2. From each batch of cells, the first image of the pressing tools is taken to determine the
    transformation matrix, which is used in the next step to transfrom the warped image into a
    straingt rectangle. This transformation matrix assignes a conversion factor from pixel to mm,
    which enables precise distance measurements in millimeters ensuring a consistent aspect ratio.
    3. All images are then transformed accordingly and split into image sections to be able to
    assign the alignment of each cell component later.
    4. For some components like the anode, the images are preprocessed for better detection.
    Preprocessing steps whicha are applied are increased contrast and a 2D convolution with an
    edge detection kernel.
    5. The circles of all components are detected with OpenCV HoughCircles. The offset to the
    pressing tool center is used as a reference to determine how much each component is misaligned
    from the center of the cell in x and y. The x-axis is from the left to the right, while the
    y-axis is from the top to the bottom (as it is specified from the image array in pixel)
    6. To account for the thickness of the parts a correction factor is determined using the height
    and the shift of the center of the pressing tool (Limit/Lina Scholz/Python Scripts/thickness_correction_factor).
    This correction factor is applied according to the thickness of the parts and the pressing tool
    position.
    7. The output is a JSON file, and an image stacked all image sections. The JSON file provides
    information about the aligment, image settings and calibration. The alignment is given for each
    cell in pixel and in mm, where the alignment in mm is already corrected to account for the
    thickness. Further the cell, step, presing tool position, radius, sample_ID and the row and
    column within the large stacked image are given. The image settings provide the information
    about the subsize of the image sections, cell number, step, mm_to_pixel and the filename. The
    calibration gives a list about the correction factor for each pressing tool in x and y and the
    thickness of the parts.

Important to note:
    Parameters which are used for fine tuning and optimization of the detection are:
    - r_parts: defines the range of the radius of the circle to be detected
    - hough_params (param1, param2): param1 and param2 are variables which can be set within the
      HoughCircle function in OpenCV; param1 defines how many edges wil be detected, which affects if
      faint circles with weak gradients are detected; param2 defines the threshold if a detected
      circle is valid, therefore higher values result in stricter cirteria for the detecion and fewer circles
    - mm_to_pixel: defining the pixel resolution in the image where the circles are to be detected
    Parameters which are subject to change are:
    - r_params: if order or size of components changes
    - z_thickness: if order or size of parts changes

Usage: The input it the filename where the images are stores. This does not have to be specified as
       it is automatically set by taking the run_ID from the database, which then specifies the
       folder directly without any input needed. The script will then output a JSON file and a
       stacked image with all images of each cell and step.

"""

import h5py
import os
import cv2
import re
import json
import numpy as np
import pandas as pd
from PIL import Image
from scipy import signal
import sqlite3
from pathlib import Path


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

def _detect_ellipses(img: np.array, r: tuple) -> tuple[list[list], np.array]:
    """ Takes image, detects ellipses of pressing tools and provides list of coordinates.

    Args:
        img (array): image array

    Return:
        coords_ellipses (list[list]): list with all six center coordinates of pressing tools
    """
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

def _detect_circles(img: np.array, radius: tuple, params: tuple) -> tuple[list[list], list[list], np.array]:
    """ Takes image, detects circles of compoments and provides list of coordinates.

    Args:
        img (array): image array
        radius (tuple): (minimum_radius, maximum_radius) to detect
        params (tuple): (param1, param2) for HoughCircles

    Return:
        coords_circles (list[list]): list with all center coordinates of components
    """
    # Apply Hough transform
    detected_circles = cv2.HoughCircles(img,
                    method = cv2.HOUGH_GRADIENT,
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
            cv2.circle(img, (a, b), r, (255, 255, 255), 2) # Draw the circumference of the circle
            cv2.circle(img, (a, b), 1, (255, 255, 255), 5) # Show center point drawing a small circle
    else:
        coords_circles = None
        r_circles = None
    return coords_circles, r_circles, img

def _convolution(image: np.array, filter: np.array) -> np.array:
    """ Takes image an convolutes it with the given filter
    """
    image_convolved = signal.convolve2d(image, filter, boundary='symm', mode='same')
    # Compute the magnitude for the case of complex kernels
    image_convolved = np.abs(image_convolved)
    # Normalize the magnitude to the range [0, 255] and convert to uint8
    image_normalized = cv2.normalize(image_convolved, None, 0, 255, cv2.NORM_MINMAX)
    image_normalized = image_normalized.astype(np.uint8)
    return image_normalized

def _preprocess_image(image: np.array, step: int) -> np.array:
    """ Takes image and applies preprocessing steps (blur, contrast)

    Args:
        image (array): image array
        step (int): robot assembly step

    Return:
        processed_image (array): processed image
    """
    if step == 2:
        image_contrast = cv2.convertScaleAbs(image, alpha=2.5, beta=0) # contrast
        # a kernel with imaginary numbers gave the best results
        # Horizontal operator is real, vertical is imaginary (for the gradients in the image)
        filter_2 = np.array([[-3-3j, 0-10j, +3-3j], [-10+0j, 0+0j, +10+0j], [-3+3j, 0+10j, +3+3j]])
        processed_image = _convolution(image_contrast, filter_2)
    else:
        processed_image = image # no preprossessing
    return processed_image



class ProcessImages:
    def __init__(self, path):
        # TRANSFORMATION ---------------------------------------------------------------------------
        self.path = path # path to images
        self.run_ID = Path(self.path).name  # get run_ID from path
        self.ref = [] # list with references (coords and corresponding cell numbers)
        self.data_list = [] # list to store image data
        self.df = pd.DataFrame(columns=["cell", "step", "press", "array"]) # data frame for all data
        # coordinates of pressing tools in mm
        self.press_position = [[0, 0], [0, 100], [95, 0], [95, 100], [190, 0], [190, 100]] # sorted by press position
        self.mm_coords = np.float32([[0, 0], [190, 0], [190, 100], [0, 100]]) # pressing tools in edges

        # Parameter, which are subject to change (of whole camera setup) ---------------------------
        self.mm_to_pixel = 10
        self.offset_mm = 20 # mm
        self.r = (21.0, 22.8) # (min, max) radius of pressing tool for reference detection [mm]
        self.r_ellipse = (20.5, 24.0) # (min, max) radius of pressing tool for reference detection [mm]

        # ALIGNMENT --------------------------------------------------------------------------------
        self.alignment_df = pd.DataFrame() # storing alignment in data frame

        # Parameter, which are subject to change (of whole camera setup) ---------------------------
        # radius of all parts from cell in mm (key corresponds to step)
        self.r_part = {0: (9.75, 10.25), 1: (9.75, 10.25), 2: (7.25, 7.75), 3: (7, 8), 4: (7.75, 8.25),
                       5: (7.75, 8.25), 6: (6.75, 7.25), 7: (7.55, 8.25), 8: (6.75, 7.7), 9: (7.5, 8.5),
                       10: (7.5, 8.5)}
        # parameter for HoughCircles (param1, param2)
        self.hough_params =[(30, 50), (30, 50), (5, 10), (30, 50), (30, 50),
                      (30, 50), (5, 25), (30, 50), (5, 20), (30, 50), (30, 50)]
        # parameter to account for thickness of parts and correct center accordingly
        self.z_correction = [(-0.175, -0.33), (-0.175, -0.2), # dz/dx & dz/dy values
                             (0.0375, -0.33), (0.0375, -0.2),
                             (0.125, -0.33), (0.125, -0.2)] # mm thickness to mm x,y shift
        # thickness of the stack for each assembly step in mm
        self.z_thickness = [0, 2.7, 0.3, 0.3, 1.55, 1.55, 1.55, 2.55, 3.3, 3.5, 3.5]

    def _get_references(self, filenameinfo: list[dict], img: np.array, ellipse_detection=True) -> tuple[np.array, list]:
        """ Takes each image from step 0 and gets the four corner coordinates of the pressing tools

        Args:
            filenameinfo (list[dicts]): list of dicts with press, cell, step
            img (array): image array
            ellipse_detection (bool): True if circles should be detected to find reference coordinates

        Returns:
            tuple with transformation matrix and list of cell numbers
        """
        img = cv2.convertScaleAbs(img, alpha=2, beta=0) # increase contrast
        ref_image_name = "_".join(str(d["c"]) for d in filenameinfo) # name with all cells belonging to reference

        if ellipse_detection:
            r_ellipse = tuple(rpx * self.mm_to_pixel for rpx in self.r_ellipse)
            coordinates, _, image_with_circles = _detect_ellipses(img, r_ellipse)
        else:
            r_circle = tuple(rpx * self.mm_to_pixel for rpx in self.r)
            coordinates, _, image_with_circles = _detect_circles(img, r_circle)

        # Draw all detected ellipses and save image to check quality of detection
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/reference"):
            os.makedirs(self.path + "/reference")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/reference/{ref_image_name}.jpg", image_with_circles)

        transformation_M = self._get_transformation_matrix(coordinates) # determine trasnformation matrix
        return (transformation_M, [d["c"] for d in filenameinfo]) # transformation matrix with cell numbers

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

    def _transform_split(self, img: np.array, m: np.array, filename: str) -> dict[np.array]:
        """ Transform and crop an image to give one sub-image per press position.

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
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/transformed"):
            os.makedirs(self.path + "/transformed")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/transformed/{filename.split(".")[0]}.jpg", transformed_image)
        # Crop the image
        cropped_images = {}
        for i, c in enumerate(self.press_position):
            # set zero in case it gives a negative number
            # set to maximum width, height in case of too large number
            height, width = img.shape
            bottom_right_y = min((c[1] + 2*self.offset_mm) * self.mm_to_pixel, height)
            bottom_right_x = min((c[0] + 2*self.offset_mm) * self.mm_to_pixel, width)
            top_left_y = max(bottom_right_y - 2*self.offset_mm*self.mm_to_pixel, 0)
            top_left_x = max(bottom_right_x - 2*self.offset_mm*self.mm_to_pixel, 0)
            cropped_image = transformed_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            cropped_images[i+1] = cropped_image
        return cropped_images

    def _thickness_correction(self, p: int, s: int, center: tuple) -> tuple:
        """ Correct center position for component height.

        Args:
            p (int): pressing tool position
            s (int): step
            center (tuple): coordinates (x, y)

        Returns:
            (x_corr, y_corr) (tuple): corrected coordinates
        """
        position = p - 1 # index to 0 to find in list
        if (s >= 1) & (s < 4): # account for bottom part
            x_corr = center[0] - self.z_thickness[s] * self.z_correction[position][0]
            y_corr = center[1] - self.z_thickness[s] * self.z_correction[position][1]
        elif (s >= 4) & (s < 7): # account for separator
            x_corr = center[0] - self.z_thickness[s] * self.z_correction[position][0]
            y_corr = center[1] - self.z_thickness[s] * self.z_correction[position][1]
        elif s >= 7: # account for spacer
            x_corr = center[0] - self.z_thickness[s] * self.z_correction[position][0]
            y_corr = center[1] - self.z_thickness[s] * self.z_correction[position][1]
        else: # no thickness to correct for
            x_corr = center[0]
            y_corr = center[1]

        return (x_corr, y_corr)

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
                info = _parse_filename(filename) # extract info from filename
                if all(d["s"] == 0 for d in info): # if step 0, get reference coordinates
                    matrix = self._get_references(info, content) # transformation matrix with cell numbers
                    self.ref.append(matrix)
                self.data_list.append((filename, info, content)) # store info and image array
        return self.data_list

    def store_data(self, data_list: list[tuple]) -> pd.DataFrame:
        """ For each image array transform image and store image sections in DataFrame.

        Returns:
            self.df (DataFrame): columns cell, step, press, transformed image section, center coordinates
        """
        for name, information, image in data_list:
            for array, numbers in self.ref:
                if numbers == [d["c"] for d in information]: # find matching transformation matrix for cell numbers
                    transformation_matrix = array
            image_sections = self._transform_split(image, transformation_matrix, name) # trasnform and split image
            for dictionary in information:
                # add information to data frame
                row = [dictionary["c"], dictionary["s"], dictionary["p"], image_sections[int(dictionary["p"])]]
                self.df.loc[len(self.df)] = row

        # save images in one big stacked image
        self.height, self.width = self.df["array"][0].shape[:2]
        self.df = self.df.sort_values(by=["cell", "step"]) # Ensure images are sorted by 'cell' and 'step'
        # Create a 10x36 grid composite image
        image_rows = []
        cols = []
        rows = []
        for i, cell in enumerate(self.df["cell"].unique()):
            cell_images = self.df[self.df["cell"] == cell].sort_values(by="step")["array"].to_list()
            cell_img = cell_images/np.max(cell_images)*255 # convert to 8 bit
            cell_img = cell_img.astype(np.uint8) # image array
            row_image = np.hstack(cell_img)  # Concatenate images in one row
            image_rows.append(row_image)
            rows.extend([i] * len(cell_images))
            cols.extend(range(len(cell_images)))
        composite_image = np.vstack(image_rows)  # Stack all rows vertically
        self.df["img_row"] = rows
        self.df["img_col"] = cols
        # create path if not existent
        data_dir = os.path.join(self.path, "json")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # Save as .h5
        h5_filename = os.path.join(data_dir, f"alignment.{self.run_ID}.h5")
        with h5py.File(h5_filename, "w") as h5_file:
            h5_file.create_dataset("image", data=composite_image)
        # Save as .jpg
        jpg_filename = os.path.join(data_dir, f"alignment.{self.run_ID}.jpg")
        Image.fromarray(composite_image).save(jpg_filename)

        return self.df

    def get_centers(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Detect centers of parts for each image section in data frame.

        Returns:
            self.df (data frame): data frame with column of center coordinates added
        """
        x = [] # list to store coordinates
        y = []
        radius = [] # list to store radius
        for index, row in df.iterrows():
            # get radius range of component
            r = tuple(int(x * self.mm_to_pixel) for x in self.r_part[row["step"]])
            img = _preprocess_image(row["array"], row["step"]) # preprocess image
            parameter = self.hough_params[row["step"]] # parameter for HoughCircles
            if row["step"] == "type in step of part which should be detected as ellipse":
                center, rad, image_with_circles = _detect_ellipses(img, r, parameter)
            else: # detect circle
                center, rad, image_with_circles = _detect_circles(img, r, parameter)
            # Assuming center as a list containing a tuple
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
            # if folder doesn't exist, create it
            if not os.path.exists(self.path + "/detected_circles"):
                os.makedirs(self.path + "/detected_circles")
            # Save the image with detected circles
            filename = f"c{row["cell"]}_p{row["press"]}_s{row["step"]}"
            cv2.imwrite(self.path + f"/detected_circles/{filename}.jpg", image_with_circles)
        # store raw coordinates in pixel
        df["x"] = x
        df["y"] = y
        df["r_mm"] = radius
        # get difference to pressing tool in pixel
        df = df.sort_values(by=["cell", "step"]) # Ensure images are sorted by 'cell' and 'step'
        dx_px_list = []
        dy_dx_list = []
        for cell in df["cell"].unique():
            cell_df = df[df["cell"] == cell]
            dx_px = (cell_df["x"] - cell_df.loc[cell_df['step'] == 0, 'x'].iloc[0]).tolist()
            dy_px = (cell_df["y"] - cell_df.loc[cell_df['step'] == 0, 'y'].iloc[0]).tolist()
            dx_px_list.extend(dx_px)
            dy_dx_list.extend(dy_px)
        df["dx_px"] = dx_px_list
        df["dy_px"] = dy_dx_list

        # get difference to pressing tool in mm
        df["dx_mm"] = df["dx_px"] / self.mm_to_pixel
        df["dy_mm"] = df["dy_px"] / self.mm_to_pixel

        self.df = df
        return df

    def correct_for_thickness(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Account for thickness of parts, correcting corresponding distortion in coordinates.

            From the reference image it is determined, how much the hight of the parts move the
            center of the parts in the different pressing tool positions due to the angle of the
            camera. This is determined manually.
        """
        x_corrected = [] # lists to store data
        y_corrected = []
        for index, row in df.iterrows():
            position = row["press"]
            step = row["step"]
            # apply thickness correction
            coords_corrected = self._thickness_correction(position, step, (row["dx_mm"], row["dy_mm"]))
            x_corrected.append(coords_corrected[0])
            y_corrected.append(coords_corrected[1])
        # store in data frame
        df["dx_mm_corr"] = x_corrected
        df["dy_mm_corr"] = y_corrected
        df["dz_mm_corr"] = np.sqrt(df["dx_mm_corr"]**2 + df["dy_mm_corr"]**2).round(5)
        self.df = df
        return df

    def save(self) -> pd.DataFrame:
        """ Save data with all coordinates, radii and alignments.
        """
        # add sample ID
        sample_IDs = [self.run_ID + "_" + f"{num:02}" for num in self.df["cell"]]
        # uncomment if special sample_ID
        # sample_IDs = [f"241022_{self.run_ID}_2-13_{num:02}" if num < 14
        #               else f"241023_{self.run_ID}_14_36_{num:02}" for num in self.df["cell"]]
        self.df["sample_ID"] = sample_IDs

        # Building JSON structure and save it
        json_data = {
            "alignment": self.df[["cell", "step", "press", "r_mm",
                                  "dx_px", "dy_px", "dx_mm_corr", "dy_mm_corr",
                                  "img_row", "img_col", "sample_ID"]].to_dict(orient="records"),
            "img_settings": {
                "subsize": (self.width, self.height),
                "cells": len(self.df["cell"].unique()),
                "steps": len(self.df["step"].unique()),
                "mm_to_px": self.mm_to_pixel,
                "raw_filename": f"alignment.{self.run_ID}.h5",
                "comp_filename": f"alignment.{self.run_ID}.jpg"
            },
            "calibration": {
                "dxdz": [x[0] for x in self.z_correction],
                "dydz": [x[1] for x in self.z_correction],
                "step_z_mm": self.z_thickness,
            },
        }
        # Write JSON to file
        data_dir_json = os.path.join(self.path, "json")
        json_path = os.path.join(data_dir_json, f"alignment.{self.run_ID}.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)

        # save as excel
        data_dir_data = os.path.join(self.path, "data")
        self.df = self.df.drop(columns=["array"]) # save without image array
        if not os.path.exists(data_dir_data):
            os.makedirs(data_dir_data)
        with pd.ExcelWriter(os.path.join(data_dir_data, "data.xlsx")) as writer:
            self.df.to_excel(writer, sheet_name='coordinates', index=False)

        return self.df


if __name__ == '__main__':

    # Get Run ID from database
    DATABASE_FILEPATH = "C:/Modules/Database/chemspeedDB.db"
    with sqlite3.connect(DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT `value` FROM Settings_Table WHERE `key` = 'Base Sample ID'")
        run_id = cursor.fetchone()[0]

    # PARAMETER
    IMAGE_FOLDER = "C:/Aurora_images/"
    folderpath = os.path.join(IMAGE_FOLDER, run_id)

    obj = ProcessImages(folderpath)
    data_list = obj.load_files()
    df = obj.store_data(data_list)
    df = obj.get_centers(df)
    df = obj.correct_for_thickness(df)
    coordinates_df = obj.save()

