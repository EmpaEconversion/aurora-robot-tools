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

class ProcessData:
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
        self.r = (210, 228) # (min, max)
        self.r_ellipse = (205, 240) # (min, max)

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
        img = cv2.convertScaleAbs(img, alpha=2, beta=0) # increase contrast
        ref_image_name = "_".join(str(d["c"]) for d in filenameinfo) # name with all cells belonging to reference

        if ellipse_detection:
            coordinates = self._detect_ellipses(img)
        else:
            coordinates = self._detect_circles(img)

        # hull = ConvexHull(coordinates) # Compute the convex hull
        # ref_coords = [coordinates[i] for i in hull.vertices] # Extract the corner points
        # Draw all detected ellipses and save image to check quality of detection
        resized_img = cv2.resize(img, (1200, 800)) # Set image size
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/reference"):
            os.makedirs(self.path + "/reference")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/reference/{ref_image_name}.jpg", resized_img)

        transformation_M = self._get_transformation_matrix(coordinates)
        return (transformation_M, [d["c"] for d in filenameinfo]) # transformation matrix with cell numbers

    def _detect_ellipses(self, img: np.array) -> list[list]:
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
                    coords_ellipses.append((cx, cy))
        return coords_ellipses

    def _detect_circles(self, img):
        """ Takes image, detects circles of pressing tools and provides list of coordinates.

        Args:
            img (array): image array
        Return:
            coords_circles (list[list]): list with all six center coordinates of pressing tools
        """
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
        coords_circles = [] # list to store reference coordinates
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for circle in detected_circles[0, :]:
                coords_circles.append((circle[0], circle[1]))
        # Draw all detected circles and save image to check quality of detection
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img, (a, b), r, (0, 0, 255), 10) # Draw the circumference of the circle
            cv2.circle(img, (a, b), 1, (0, 0, 255), 10) # Show center point drawing a small circle
        return coords_circles

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
        median_y = sorted(y_values)[len(y_values) // 2] # Calculate the median of the y-values
        # Split the list based on the median y-value
        lower_y_group = [center for center in centers if center[1] < median_y]
        higher_y_group = [center for center in centers if center[1] >= median_y]
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
        # for cross check save image:
        resized_img = cv2.resize(transformed_image, (1200, 800)) # Set image size
        # if folder doesn't exist, create it
        if not os.path.exists(self.path + "/transformed"):
            os.makedirs(self.path + "/transformed")
        # Save the image with detected ellipses
        cv2.imwrite(self.path + f"/transformed/{filename}.jpg", resized_img)
        # Crop the image
        cropped_images = {}
        for i, c in enumerate(self.press_position):
            top_left_y = (c[1] + self.offset_mm) * self.mm_to_pixel
            top_left_x = (c[0] + self.offset_mm) * self.mm_to_pixel
            bottom_right_y = (c[1] + 2*self.offset_mm) * self.mm_to_pixel
            bottom_right_x = (c[0] + 2*self.offset_mm) * self.mm_to_pixel
            cropped_image = transformed_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
            # for cross check save image:
            resized_img = cv2.resize(cropped_image, (1200, 800)) # Set image size
            # if folder doesn't exist, create it
            if not os.path.exists(self.path + "/image_sections"):
                os.makedirs(self.path + "/image_sections")
            # Save the image with detected ellipses
            cv2.imwrite(self.path + f"/image_sections/{filename}.jpg", resized_img)
            cropped_images[i] = cropped_image
        return cropped_images

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
        return self.data_list

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
            for num, dictionary in enumerate(information):
                row = [dictionary["c"], dictionary["s"], dictionary["p"], image_sections[num]]
                self.df.loc[len(self.df)] = row
        return self.df

#%% RUN CODE
if __name__ == '__main__':

    # PARAMETER
    folderpath = "C:/test"

    obj = ProcessData(folderpath)
    images_list = obj.load_files()
    image_info_df = obj.store_data()

print(image_info_df)
