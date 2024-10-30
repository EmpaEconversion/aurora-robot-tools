"""
Script to read in images from folder and transform warped rectangle in straight rectangle

"""

import h5py
import math
import os
import numpy as np
import cv2

#%% CLASS WITH FUNCTIONS

class TRANSFORM:
    def __init__(self, folderpath):
        self.folderpath = folderpath # path to images
        self.savepath = folderpath + "/processed" # path to save information
        self.transformed_images = [] # list to store transformed image arrays
        self.data_list = [] # list to store image data
        self.reference = [] # list to store images from pressing tool (step 0)
        self.ref = [] # list to store reference coordinates
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)

        # Parameter which might need to be changes if camera position changes
        self.r = (210, 228) # (min, max)
        self.r_ellipse = (205, 240) # (min, max)

    def _parse_filename(filename: str) -> dict: # TODO
        """ Takes photo filename and returns dict of lists of press cell and step

        Longer description blah blah blah

        Args:
            filename (str): the filename of the photo used

        Returns:
            dict: dictionary containing keys 'step' 'press' 'cell' with lists of numbers
        """

    # read files (images) from folder ----------------------------------------------
    def load_files(self) -> list:
        """ Loads images and stores them in list with filename and image array

        All images are loaded and stored in list with a tuple of their filename and 8 bit image
        array. Images from step 0 are stored additionally in a separate list to use them as a
        reference for the coordinate transformation.

        Returns:
            list: list containing filename and image array
        """
        print("load files")
        for filename in os.listdir(self.folderpath):
            if filename.endswith('.h5'): # read all .h5 files
                filepath = os.path.join(self.folderpath, filename)
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

    # get reference coordinates ----------------------------------------------
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

    # transform warped rectangle to straight rectangle ----------------------------------------------
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

#%% RUN CODE
if __name__ == '__main__':
    # PARAMETER
    # path to files
    path = 'G:/Limit/Lina Scholz/robot_files_gen14'

    # EXECUTE
    obj = TRANSFORM(path)
    files = obj.load_files() # load all images from folder
    reference = obj.get_reference() # get center points from first step of assembly as reference
    transformed = obj.transform_pixel_coordinates() # transform pixel coordinates to get "rectangular shape"
