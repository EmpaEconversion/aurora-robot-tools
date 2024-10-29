'''
Describe purpose of Script
'''

import h5py
import numpy as np
import pandas as pd
import cv2

#%% PLOT IMAGE FROM WHICH DISTORTION IS DETERMINED

# Change image size
desired_width = 1200
desired_height = 800

# Load image from .h5 file
with h5py.File("G:/Limit/Lina Scholz/robot_files_gen14/transformed/p01c01s0_p03c02s0_p05c03s0_p02c04s0_p04c05s0_p06c06s0.h5", 'r') as f: 
    content = f['image'][:]
    # In 8-Bit umwandeln
    content = content / np.max(content) * 255
    img = content.astype(np.uint8)

img = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # 1.5

img = cv2.GaussianBlur(img, (5, 5), 2) # Apply a Gaussian blur to the image before detecting circles (to improve detection)
# Apply Hough transform
detected_circles = cv2.HoughCircles(img,  
                    cv2.HOUGH_GRADIENT, 
                    dp = 1, 
                    minDist = 100, 
                    param1 = 30, param2 = 50, 
                    minRadius = 520, maxRadius = 580)  

print(detected_circles)

if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

if detected_circles is not None:
    # Draw all detected circles
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 0, 255), 10)

        # Show center point drawing a small circle
        cv2.circle(img, (a, b), 1, (0, 0, 255), 10)

        # Print the center points of the detected circles
        print(f"Detected Circle Center: ({a}, {b}), Radius: {r}")

resized_img = cv2.resize(img, (desired_width, desired_height))

# Show image with detected images
cv2.imshow("Detected Circles", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Store image
# cv2.imwrite('G:/Limit/Lina Scholz/Python Script/Test images/z-distortion(3).jpg', img)

#%% EXTRACTED VALUES TO OFFSET

# 1mm thickness corresponds to ... pixel:
pos1 = (3, 6) # pixel
pos2 = (0, 6)
pos3 = (2.5, 6)

pos4 = (3, 3)
pos5 = (0, 3)
pos6 = (2.5, 3)

positions = [pos1, pos2, pos3, pos4, pos5, pos6] # pixel

#%% CORRECT COORDINATE DATA BASE WITH OFFSET (bottom part, separator, spacer)

# thickness in mm
t_s1 = 0.3 # thickness bottom part 
t_s4 = 1.25 # thickness separator
t_s7 = 1 # thickness spacer

# from step 1 onwards correct with:
corr_s1 = [tuple(round(x * t_s1, 3) for x in pos) for pos in positions]
print(corr_s1)

# from step 4 onwards correct with:
corr_s4 = [tuple(round(x * (t_s1+t_s4), 3) for x in pos) for pos in positions]
print(corr_s4)

# from step 7 onwards correct with:
corr_s7 = [tuple(round(x * (t_s1+t_s4+t_s7), 3) for x in pos) for pos in positions]
print(corr_s7)