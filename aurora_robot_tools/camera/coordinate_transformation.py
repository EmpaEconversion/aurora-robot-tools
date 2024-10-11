

import h5py
import math
import os
import numpy as np
from operator import itemgetter
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#%% CLASS WITH FUNCTIONS

class TRANSFORM:
    def __init__(self, folderpath, savepath):
        self.savepath = savepath
        self.folderpath = folderpath

    # read files (images) from folder
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
                        if int(filename.split("_")[3]) == 0: # only load images from first step as reference
                            self.reference.append((filename, content))
                        self.data_list.append((filename, content))
                    except:
                        print(f"wrong file name: {filename}")
        return self.data_list
    
    # get reference coordinates
    def get_reference(self):
        print("get reference coordinates")
        self.ref = []
        for name, img in self.reference:
            img = cv2.GaussianBlur(img, (9, 9), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
                # Apply Hough transform
            detected_circles = cv2.HoughCircles(img,  
                            cv2.HOUGH_GRADIENT, 
                            dp = 1, 
                            minDist = 100, 
                            param1 = 30, param2 = 50, 
                            minRadius = 210, maxRadius = 265) # HARD CODED!!!
            
            # Extract center points and their pressing tool position
            coords = []
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))
                for circle in detected_circles[0, :]:
                    center = (circle[0], circle[1])
                    coords.append(center)
            self.ref.append((name, coords))
        return self.ref

    # transform warped rectangle to straight rectangle ----------------------------------------------
    def transform_pixel_coordinates(self):
        print("transform warped image in pixel coordinates")
        transformed_images = []
        for name, img in self.data_list:
            batch = int(name.split("_")[5].split(".")[0])
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

            # Save the transformed_image to an HDF5 file
            p = self.savepath + f"/ {name.split(".")[0]}_transformed.h5"
            with h5py.File(p, 'w') as h5_file:
                h5_file.create_dataset('transformed', data=transformed_image)

        return transformed_images

#%% RUN CODE

# PARAMETER
# path to files
path = 'G:/Limit/Lina Scholz/robot_files_20241004'
# path to store transformed images
if not os.path.exists('G:/Limit/Lina Scholz/robot_files_20241004/transformed'):
    os.makedirs('G:/Limit/Lina Scholz/robot_files_20241004/transformed')
savepath = 'G:/Limit/Lina Scholz/robot_files_20241004/transformed'

# EXECUTE
obj = TRANSFORM(path, savepath)
files = obj.load_files() # load all images from folder
reference = obj.get_reference() # get center points from first step of assembly as reference
transformed = obj.transform_pixel_coordinates() # transform pixel coordinates to get "rectangular shape"








