"""
Script to read in images from folder and detect circles and their alignment

"""

import cv2
import os
import math
import numpy as np
import pandas as pd
import h5py
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#%% CLASS WITH FUNCTIONS

class ALIGNMENT:
    def __init__(self, path):
        self.path = path
        self.data_list = []
        self.df_images = pd.DataFrame(columns=["pos", "cell", 
                                               "s0_coords", "s1_coords", "s2_coords", "s3_coords", "s4_coords", "s5_coords", "s6_coords", "s7_coords", "s8_coords", "s9_coords", "s10_coords",
                                               "s0_r", "s1_r", "s2_r", "s3_r", "s4_r", "s5_r", "s6_r", "s7_r", "s8_r", "s9_r", "s10_r",
                                               "s1_align", "s2_align", "s4_align", "s5_align", "s6_align", "s7_align", "s8_align", "s9_align", "s10_align",
                                               "s1_align [mm]", "s2_align [mm]", "s4_align [mm]", "s5_align [mm]", "s6_align [mm]", "s7_align [mm]", "s8_align [mm]", "s9_align [mm]", "s10_align [mm]"])
        # 0:pressing tool, 1:bottom part, 2:anode, 4:separator, 5:electrolyte, 6:cathode, 7:spacer, 8:spring, 9:top part, 10:after pressing
        self.r_min = {0: 200, 1: 200, 2: 145, 4: 160, 5: 170, 6: 140, 7: 150, 8: 140, 9: 140, 10: 160} 
        self.r_max = {0: 230, 1: 230, 2: 175, 4: 183, 5: 190, 6: 162, 7: 178, 8: 170, 9: 185, 10: 190} 

    # read files in list ----------------------------------------------
    def read_files(self):
        print("\n read files in list")
        for filename in os.listdir(self.path):
            if filename.endswith('.h5'):
                filepath = os.path.join(self.path, filename)
                with h5py.File(filepath, 'r') as f:
                    content = f['image'][:]
                    # convert to 8 bit
                    content = content/np.max(content)*255
                    content = content.astype(np.uint8)
                    try:
                        self.data_list.append((filename, content))
                    except:
                        print(f"wrong file name: {filename}")
        return self.data_list 
    
    # get circle coordinates ----------------------------------------------
    def get_coordinates(self):
        print("\n detect circles and get circle coordinates")
        positions = [] # create lists & dictionaries to store informaiton
        cell_numbers = []
        coordinates = {0: [], 1: [], 2: [], 4: [], 5:[], 6: [], 7: [], 8: [], 9: [], 10:[]} # stores coordinates for all steps
        radius = {0: [], 1: [], 2: [], 4: [], 5:[], 6: [], 7: [], 8: [], 9: [], 10:[]} # stores radius for all steps

        for name, img in self.data_list:
            try:
                step = int(name.split("_")[0].split("s")[1])
            except:
                step = int(name.split(".")[0].split("s")[1])
                print(f"fewer cells or wrong filename (check folder with files and their names): {name}")
            if step == 0: # assign position and cell number in data frame
                img = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # increase contrast
                img = cv2.GaussianBlur(img, (5, 5), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
                string = name.split(".")[0]
                for i in range(len(string.split("_"))):
                    if len(string.split("_")) > 1:
                        positions.append(str(string.split("_")[i].split("c")[0][-2:]))
                        cell_numbers.append(str(string.split("_")[i].split("c")[1].split("s")[0]))
                    else:
                        print("only one cell in pressing tools")
                        positions.append(str(string.split("c")[0][-2:]))
                        cell_numbers.append(str(string.split("c")[1].split("s")[0]))
            elif step == 2: # increase constrast for the anode
                img = cv2.convertScaleAbs(img, alpha=2, beta=0) 
                img = cv2.GaussianBlur(img, (5, 5), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
            elif step == 6: # no contrast change for cathode
                img = cv2.GaussianBlur(img, (5, 5), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
            elif step == 8:
                img = cv2.convertScaleAbs(img, alpha=1.25, beta=0) # increase contrast
                # no increased contrast for spring?? # TODO
                # no gaussian blur for spring
            else:
                img = cv2.convertScaleAbs(img, alpha=1.25, beta=0) # increase contrast
                img = cv2.GaussianBlur(img, (5, 5), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)

            # Apply Hough transform
            detected_circles = cv2.HoughCircles(img,  
                            cv2.HOUGH_GRADIENT, 
                            dp = 1, 
                            minDist = 100, 
                            param1 = 30, param2 = 50, 
                            minRadius = self.r_min[step], maxRadius = self.r_max[step]) 
            
            # Extract center points and their pressing tool position
            coords_buffer_list = [[0, 0]] * 6
            r_buffer_list = [0] * 6
            if detected_circles is not None:
                detected_circles = np.uint16(np.around(detected_circles))
                for circle in detected_circles[0, :]:
                    # assign circle pressing tool
                    # constrain to avoid too many circles
                    if (circle[1] > 2800) & (circle[1] < 3050):
                        if (circle[0] < 650) & (circle[0] > 350): # position 4
                            coords_buffer_list[3]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[3] = circle[2] # radius
                        elif (circle[0] > 2600) & (circle[0] < 2760): # position 5
                            coords_buffer_list[4]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[4] = circle[2] # radius
                        elif (circle[0] > 4600) & (circle[0] < 5100): # position 6
                            coords_buffer_list[5]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[5] = circle[2] # radius
                        else: 
                            print(f"\n circle in lower row couldnt be assigned: ({circle[0]}, {circle[1]})")
                            # Create a mask that identifies incorrectly positioned circles to be remove
                            mask = ~np.all(np.isin(detected_circles, circle), axis=-1)  # axis=-1 to compare along the last dimension
                            detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape

                    elif (circle[1] < 800) & (circle[1] > 650):
                        if (circle[0] < 600) & (circle[0] > 400): # position 1
                            coords_buffer_list[0]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[0] = circle[2] # radius
                        elif (circle[0] > 2600) & (circle[0] < 2760): # position 2
                            coords_buffer_list[1]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[1] = circle[2] # radius
                        elif (circle[0] > 4600) & (circle[0] < 5100): # position 3
                            coords_buffer_list[2]= [circle[0], circle[1]]  # (x, y) coordinates
                            r_buffer_list[2] = circle[2] # radius
                        else: 
                            print(f"\n circle in upper row couldnt be assigned: ({circle[0]}, {circle[1]})")
                            # Create a mask that identifies incorrectly positioned circles to be remove
                            mask = ~np.all(np.isin(detected_circles, circle), axis=-1)  # axis=-1 to compare along the last dimension
                            detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape
                    else:
                        print(f"\n circle couldnt be assigned for any pressing tool: ({circle[0]}, {circle[1]})") 
                        # Create a mask that identifies incorrectly positioned circles to be remove
                        mask = ~np.all(np.isin(detected_circles, circle), axis=-1)  # axis=-1 to compare along the last dimension
                        detected_circles = np.array(detected_circles[mask]).reshape(1, -1, 3)  # Reshape

                # Draw all detected circles and save image to check quality of detection
                for pt in detected_circles[0, :]:
                    a, b, r = pt[0], pt[1], pt[2]
                    cv2.circle(img, (a, b), r, (0, 0, 255), 10) # Draw the circumference of the circle
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 10) # Show center point drawing a small circle
                desired_width = 1200 # Change image size
                desired_height = 800
                resized_img = cv2.resize(img, (desired_width, desired_height))
                # if folder doesn't exist, create it
                if not os.path.exists(self.path + "/detected_circles"):
                    os.makedirs(self.path + "/detected_circles")
                cv2.imwrite(self.path + f"/detected_circles/{name.split(".")[0]}.jpg", resized_img) # Save the image with detected circles
            
                # add values to list to collect values
                c = coordinates[step]
                c.extend(coords_buffer_list)
                coordinates[step] = c
                r = radius[step]
                r.extend(r_buffer_list)
                radius[step] = r

        # fill values into dataframe
        for num, column in enumerate(self.df_images.columns.tolist()):
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
        self.df_images= self.df_images.drop(columns=["s3_coords", "s3_r"])
        self.df_images['pos'] = self.df_images['pos'].astype(int) # get cell number and position as integers not string
        self.df_images['cell'] = self.df_images['cell'].astype(int)
        return self.df_images

    # get alignment numbers ----------------------------------------------
    def alignment_number(self):
        print("determine alignment in pixel")
        for index, row in self.df_images.iterrows():
            x_ref = row["s0_coords"][0]
            y_ref = row["s0_coords"][1]

            pos = row["pos"].astype(int)
            # distortion_correction = [(), (), (), (), (), ()]

            for i, col_name in enumerate(self.df_images.columns.tolist()[3:12]):
                n = self.df_images.columns.tolist()[-18:][i] # column name of alignment entry
                step = n.split("_")[0].split("s")[1]
                x = int(x_ref) - int(row[col_name][0])
                y = int(y_ref) - int(row[col_name][1])
                z = round(math.sqrt(x**2 + y**2), 1) # round number to one digit
                self.df_images._set_value(index, str(n), (x, y, z))
        return self.df_images
    
    # convert pixel to mm ----------------------------------------------
    def pixel_to_mm(self):
        print("\n convert pixel values to mm")
        pixel = (sum(self.df_images["s0_r"].to_list())/len(self.df_images["s0_r"].to_list()) * 2) # pixel
        mm = 20 # mm
        pixel_to_mm = mm/pixel
        print("pixel to mm: " + str(pixel_to_mm) + " mm/pixel")
        # missalignment to mm
        for i in list(range(1, 11)):
            if i != 3:
                self.df_images[f"s{i}_align [mm]"] = [(round(x * pixel_to_mm, 3), round(y * pixel_to_mm, 3), round(z * pixel_to_mm, 3)) for x, y, z in self.df_images[f"s{i}_align"].to_list()]
        # radius to mm
        for i in list(range(0, 11)):
            if i != 3:
                self.df_images[f"s{i}_r [mm]"] = [round(r * pixel_to_mm, 3) for r in self.df_images[f"s{i}_r"].to_list()]
        return self.df_images
     

#%% RUN CODE

# PARAMETER
path = "G:/Limit/Lina Scholz/robot_files_gen14/transformed"

# EXECUTE
obj = ALIGNMENT(path)
imgages = obj.read_files() # list with all images given as a list 
images_detected = obj.get_coordinates() # get coordinates of all circles
images_alignment = obj.alignment_number() # get alignment
images_alignment_mm = obj.pixel_to_mm() # get alignment number in mm 

print(images_detected.head())
print(images_alignment.head())
print(images_alignment_mm.head())

#%% SAVE

if not os.path.exists(path + "/data"):
    os.makedirs(path + "/data")
images_alignment.to_excel(path + "/data/data.xlsx")


