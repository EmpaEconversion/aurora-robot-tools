''' Lina Scholz

Script to determine amount of shift of center points in pixel due to thickness of parts.
'''

import h5py
import numpy as np
import cv2

#%% PLOT IMAGE FROM WHICH DISTORTION IS DETERMINED

# save image?
save = False
jpg = True

# File path
path = "C:/lisc_gen14/image_sections/p03c14s8_p05c15s8_p02c16s8_p04c17s8_p06c18s8_pos_3.jpg"

if jpg:
    # Load the JPG image
    img = cv2.imread(path)
    # Convert the image to a NumPy array (if not already)
    img = np.array(img)
    # convert to 8 bit
    img = img/np.max(img)*255
    img = img.astype(np.uint8)
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

else:
    # Load image from .h5 file
    with h5py.File(path, 'r') as f:
        content = f['image'][:]
        # In 8-Bit umwandeln
        content = content / np.max(content) * 255
        img = content.astype(np.uint8)

# image processing
# img = cv2.convertScaleAbs(img, alpha=1.5, beta=0) # increase contrast
img = cv2.GaussianBlur(img, (5, 5), 2) # Gaussian blur to the image before detecting circles (to improve detection)
# Apply Hough transform
detected_circles = cv2.HoughCircles(img,
                    cv2.HOUGH_GRADIENT,
                    dp = 1,
                    minDist = 100,
                    param1 = 30, param2 = 50,
                    minRadius = 210, maxRadius = 260) # 520, 580
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

resized_img = cv2.resize(img, (700, 700))
# Show image with detected images
cv2.imshow("Detected Circles", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
if save:
    # Store image
    cv2.imwrite('G:/Limit/Lina Scholz/Python Script/Test images/z-distortion_sections.jpg', img)

#%% EXTRACTED VALUES TO OFFSET

# 1mm thickness corresponds to ... pixel:
pos1 = (15, 31.5) # pixel
pos2 = (0, 31.5)
pos3 = (13.75, 31.5)
pos4 = (15, 18.75)
pos5 = (0, 18.75)
pos6 = (13.75, 18.75)

# save in list
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
