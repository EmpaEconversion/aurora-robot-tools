
"""
Script to read in images from folder and detect circles

"""

import cv2
import os
import numpy as np
import h5py
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

#%% FUNCTIONS ------------------------------------------------------------

#%% READ IMAGES

# read files (images) from folder
def load_files_in_folder(folderpath, format, convert_bid=True, plot=False):
    data_list = []
    for filename in os.listdir(folderpath):
        if filename.endswith(format):
            filepath = os.path.join(folderpath, filename)
            with h5py.File(filepath, 'r') as f:
                content = f['image'][:]
                if convert_bid:
                    # convert to 8 bit
                    content = content/np.max(content)*255
                    content = content.astype(np.uint8)
                data_list.append((filename, content))
    if plot:
        for filename, content in data_list:
            print(f'filename: {filename}')
            print(f'content: {content[:100]}...') 
            plt.imshow(content)
            plt.axis('off')
            plt.show()

    return data_list

#%% DETECT CIRCLES IN (UNPROCESSED) IMAGE

# detect circles
def detect_circles(img, plot=True):
    # Apply a Gaussian blur to the image before detecting circles (to improve detection)
    img = cv2.GaussianBlur(img, (9, 9), 2) 
    if plot:
        # show image
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    # Apply Hough transform
    detected_circles = cv2.HoughCircles(img,  
                    cv2.HOUGH_GRADIENT, 
                    dp = 1, 
                    minDist = 100, 
                    param1 = 30, param2 = 50, 
                    minRadius = r_min, maxRadius = r_max) 
    print(detected_circles)
    print(type(detected_circles))
    
    # Extract center points
    center_points = []
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for circle in detected_circles[0, :]:
            center_points.append((circle[0], circle[1]))  # (x, y) coordinates
    center_points = np.array(center_points)  # Convert to NumPy array

    print(detected_circles)
    print("Center points:", center_points)
    return detected_circles, center_points

#%% PLOT DETECTED CIRCLES IN IMAGE 

# draw detected circles$
def plot_detected_circles(circles, img):
    # Draw circles that are detected.
    if circles is not None:
        # Apply a Gaussian blur to the image 
        img = cv2.GaussianBlur(img, (9, 9), 2) 

        # Draw all detected circles
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 0, 255), 10)

            # Show center point drawing a small circle
            cv2.circle(img, (a, b), 1, (0, 0, 255), 10)

            # Print the center points of the detected circles
            print(f"Detected Circle Center: ({a}, {b}), Radius: {r}")

    # Change image size
    desired_width = 1200
    desired_height = 800
    resized_img = cv2.resize(img, (desired_width, desired_height))

    # Save the image with detected circles
    cv2.imwrite('Detected circles/detected_circles_test.jpg', resized_img)
    # Show image with detected images
    cv2.imshow("Detected Circles", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#%% SPLIT IMAGE INTO 6 IMAGE SECTIONS

# extract images around the six detected center points at a certain patch size
def extract_images(center_points, size, plot=False):
    # Convert to list of tuples
    center_points = [tuple(x) for x in center_points.tolist()]
    
    # Create an image section for each point
    patches = [] # list to store image sections
    section_coords = []  # List to store top-left corner coordinates of each section

    for (x, y) in center_points:
        # definition of quadratic area
        x_start = max(0, x - size // 2)
        x_end = min(img_orig.shape[1], x + size // 2)
        y_start = max(0, y - size // 2)
        y_end = min(img_orig.shape[0], y + size // 2)
        
        # extract image section
        patch = img_orig[y_start:y_end, x_start:x_end]
        patches.append(patch)

        # Store all four corner points
        top_left = (x_start, y_start)
        top_right = (x_end, y_start)
        bottom_left = (x_start, y_end)
        bottom_right = (x_end, y_end)
        section_coords.append((top_left, top_right, bottom_left, bottom_right))
        
        if plot:
            # Optional: show image section
            plt.imshow(patch)
            plt.axis('off')  # Turn off axis labels
            plt.show()

        # show that coordinates are maintained
        print(f"section for the point at ({x}, {y}) from ({x_start}:{x_end}, {y_start}:{y_end})")

    return patches, section_coords

#%% DETECT CIRCLES & ELLIPSES IN IMAGE SECTIONS

# detect circles in image sections
def detect_circles_in_sections(image_sections, ellipse_detection, plot=False):

    for n, img in enumerate(image_sections):
        c = centers_array[n]

        # Check the shape of the image
        if len(img.shape) == 3:  # Color image (BGR)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img  # Already a grayscale image

        # Blur the grayscale image
        img_blur = cv2.GaussianBlur(img_gray, (9, 9), 2)

        # Store the detected circles' coordinates
        detected_circle_coords = [] 

        # Circle detection using Hough Transform
        detected_circles = cv2.HoughCircles(img_blur,  
                        cv2.HOUGH_GRADIENT, 
                        dp = 1, 
                        minDist = 100, 
                        param1 = 30, param2 = 50, 
                        minRadius = r_min, maxRadius = r_max)
        
        # Draw detected circles
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for circle in detected_circles[0, :]:
                center = (circle[0], circle[1])
                radius = circle[2]
                detected_circle_coords.append((center, radius))
                # Draw the circumference of the circle
                cv2.circle(img, center, radius, (0, 0, 255), 2)  # Red color for circles
                # Draw the center point
                cv2.circle(img, center, 2, (0, 0, 255), 3)


        if ellipse_detection:
            # Edge detection for ellipses
            edges = cv2.Canny(img_blur, 50, 150)

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
                        if 0.9 < aspect_ratio < 1.1 and r_min <= avg_radius <= r_max:
                            detected_circle_coords.append((ellipse[0], avg_radius))
                            cv2.ellipse(img, ellipse, (0, 255, 0), 2)  # Green color for ellipses

        # Print the coordinates and radii of all detected circles and ellipses
        for i, (center, radius) in enumerate(detected_circle_coords):
            print(f"Circle {i}: Center = {center}, Radius = {radius}")

        # Show the result
        if plot:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(("[" + str(c[0]) + ", " + str(c[1]) + "]"))
            plt.show()

    return detected_circle_coords

#%% PARAMETER ------------------------------------------------------------

# specify folder and format of files
folderpath = r"G:\Limit\Lina Scholz\Images Camera Adjustment\06 Images LED sanded\pickles"
format = '.h5'
# select image from folder
image_in_folder = 1
# select one of the six sections
num_image_section = 2
# set patch size for the six image sections of each pressing tool
patch_size = 650

# range of radius to detect circles
r_min = 200 # minimum radius
r_max = 250 # maximum radius

#%% RUN ------------------------------------------------------------

files = load_files_in_folder(folderpath, format, convert_bid= True) # read files (images) from folder
img_orig = files[image_in_folder][1] # select image from folder
circles_array, centers_array = detect_circles(img_orig) # detect circles
plot_detected_circles(circles_array, img_orig) # plot detected images
# img_sections, img_section_coords = extract_images(centers_array, patch_size) # extract images around the six detected center points at a certain patch size
# img_section_detected_circles_coords = detect_circles_in_sections(img_sections, ellipse_detection= False)