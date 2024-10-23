"""
Script to read in images from folder and transform warped rectangle in straight rectangle

"""

import h5py
import math
import os
import numpy as np
from operator import itemgetter
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans

#%% CLASS WITH FUNCTIONS

class TRANSFORM:
    def __init__(self, folderpath, savepath):
        self.folderpath = folderpath
        self.savepath = savepath

    # read files (images) from folder ----------------------------------------------
    def load_files(self):
        print("load files")
        self.data_list = []
        self.reference = []
        for filename in os.listdir(self.folderpath):
            if filename.endswith('.h5'):
                filepath = os.path.join(self.folderpath, filename)
                with h5py.File(filepath, 'r') as f:
                    content = f['image'][:]
                    # convert to 8 bit
                    content = content/np.max(content)*255
                    content = content.astype(np.uint8)
                    try:
                        if int(filename.split("_")[0].split("s")[1]) == 0: # only load images from first step as reference
                            self.reference.append((filename, content))
                        self.data_list.append((filename, content))
                    except:
                        if int(filename.split(".")[0].split("s")[1]) == 0: # only load images from first step as reference
                            self.reference.append((filename, content))
                        self.data_list.append((filename, content))
                        print(f"fewer cells or wrong filename (check folder with files and their names): {filename}")
        return self.data_list
    
    # get reference coordinates ----------------------------------------------
    def get_reference(self, circle_detection=False, ellipse_detection=True):
        print("get reference coordinates")
        self.ref = []
        for name, img in self.reference:
            img = cv2.convertScaleAbs(img, alpha=3, beta=0) # increase contrast

            # img = cv2.GaussianBlur(img, (9, 9), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
            
            if circle_detection:
                # Apply Hough transform
                detected_circles = cv2.HoughCircles(img,  
                                cv2.HOUGH_GRADIENT, 
                                dp = 1, 
                                minDist = 100, 
                                param1 = 30, param2 = 50, 
                                minRadius = 210, maxRadius = 228) # HARD CODED!!!
                
                # Extract center points and their pressing tool position
                coords = []
                if detected_circles is not None:
                    detected_circles = np.uint16(np.around(detected_circles))
                    for circle in detected_circles[0, :]:
                        center = (circle[0], circle[1])
                        coords.append(center)
                self.ref.append((name, coords))

                # Draw all detected circles and save image to check quality of detection
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    cv2.circle(img, (a, b), r, (0, 0, 255), 10) # Draw the circumference of the circle
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 10) # Show center point drawing a small circle
                desired_width = 1200 # Change image size
                desired_height = 800
                resized_img = cv2.resize(img, (desired_width, desired_height))
                # if folder doesn't exist, create it
                if not os.path.exists(self.savepath + "/reference_detected_circles"):
                    os.makedirs(self.savepath + "/reference_detected_circles")
                cv2.imwrite(self.savepath + f"/reference_detected_circles/{name.split(".")[0]}.jpg", resized_img) # Save the image with detected circles

            if ellipse_detection:
                coords = []
                # Edge detection for ellipses
                edges = cv2.Canny(img, 50, 150)
                # Find contours
                contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
                            if 0.9 < aspect_ratio < 1.1 and 200 <= avg_radius <= 230:
                                coords.append((ellipse[0], avg_radius))
                                cv2.ellipse(img, ellipse, (0, 255, 0), 10)  # Green color for ellipses
                                # Draw the center point
                                center = (int(ellipse[0][0]), int(ellipse[0][1]))  # Convert center coordinates to integers
                                cv2.circle(img, center, 5, (0, 255, 0), -1)
                
                # Filter out similar ellipses
                filtered_ellipses = []
                coords_ellipses = []
                for ellipse in coords:
                    (cx, cy), r = ellipse
                    # Check if the current circle is similar to any circle in the filtered list
                    if not any(np.sqrt((cx - fcx)**2 + (cy - fcy)**2) < 1 and abs(r - fr) < 1 
                            for (fcx, fcy), fr in filtered_ellipses):
                        filtered_ellipses.append(ellipse)
                        center = (cx, cy)
                        coords_ellipses.append(center)
                self.ref.append((name, coords_ellipses))

                # Draw all detected ellipses and save image to check quality of detection
                desired_width = 1200 # Change image size
                desired_height = 800
                resized_img = cv2.resize(img, (desired_width, desired_height))
                # if folder doesn't exist, create it
                if not os.path.exists(self.savepath + "/reference_detected_ellipses"):
                    os.makedirs(self.savepath + "/reference_detected_ellipses")
                cv2.imwrite(self.savepath + f"/reference_detected_ellipses/{name.split(".")[0]}.jpg", resized_img) # Save the image with detected circles

        return self.ref

    # transform warped rectangle to straight rectangle ----------------------------------------------
    def transform_pixel_coordinates(self):
        print("transform warped image in pixel coordinates")
        transformed_images = []
        batch = 0
        for name, img in self.data_list:
            height, width = img.shape[:2] # determine size of image

            ctr = self.ref[batch][1]
            ctr_sorted = [(0, 0), (0, 0), (0, 0), (0, 0)]
            for t in ctr:
                if (t[0] < 650) & (t[1] < 850):
                    ctr_sorted[0] = t
                elif (t[0] > 4750) & (t[1] < 850):
                    ctr_sorted[1] = t
                elif (t[0] > 4750) & (t[1] > 2800):
                    ctr_sorted[2] = t
                elif (t[0] < 650) & (t[1] > 2800):
                    ctr_sorted[3] = t
                else:
                    pass
            ctr_sorted = np.float32(ctr_sorted)
            # Calculate new coordinates to correct for distortion
            a1 = ctr_sorted[0][0] + ((ctr_sorted[0][1] - ctr_sorted[1][1])) / math.sin(math.atan(((ctr_sorted[0][1] - ctr_sorted[1][1])) / (ctr_sorted[1][0] - ctr_sorted[0][0])))
            a2 = ctr_sorted[0][1] + ((ctr_sorted[0][0] - ctr_sorted[3][0])) / math.sin(math.atan(((ctr_sorted[0][0] - ctr_sorted[3][0])) / (ctr_sorted[3][1] - ctr_sorted[0][1])))
            pts2 = np.float32([ctr_sorted[0], [a1, ctr_sorted[0][1]], [a1, a2], [ctr_sorted[0][0], a2]]) # corrected coordinates
            # Transform Perspective
            M = cv2.getPerspectiveTransform(ctr_sorted, pts2)
            transformed_image = cv2.warpPerspective(img, M, (width, height))
            transformed_images.append(transformed_image)

            # Save the transformed_image to jpg
            if not os.path.exists(f'{self.savepath}/transformed_jpg'):
                os.makedirs(f'{self.savepath}/transformed_jpg')
            cv2.imwrite(f'{self.savepath}/transformed_jpg/{name.split(".")[0]}.jpg', transformed_image)

            # Save the transformed_image to an HDF5 file
            p = self.savepath + f"/{name.split(".")[0]}.h5"
            with h5py.File(p, 'w') as h5_file:
                h5_file.create_dataset('image', data=transformed_image)

        return transformed_images
        

#%% RUN CODE

# PARAMETER
# path to files
path = 'G:/Limit/Lina Scholz/robot_files_gen14'
# path to store transformed images
if not os.path.exists('G:/Limit/Lina Scholz/robot_files_gen14/transformed'):
    os.makedirs('G:/Limit/Lina Scholz/robot_files_gen14/transformed')
savepath = 'G:/Limit/Lina Scholz/robot_files_gen14/transformed'

# EXECUTE
obj = TRANSFORM(path, savepath)
files = obj.load_files() # load all images from folder
reference = obj.get_reference() # get center points from first step of assembly as reference
transformed = obj.transform_pixel_coordinates() # transform pixel coordinates to get "rectangular shape"