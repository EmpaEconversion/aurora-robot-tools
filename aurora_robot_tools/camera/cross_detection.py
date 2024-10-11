

# Importing the necessary libraries 
import cv2 
import numpy as np
import matplotlib.pyplot as plt

option_1 = False
option_2 = False
option_3 = True
  
# Reading the image 
image = cv2.imread('G:/Limit/Lina Scholz/Images Camera Adjustment/06 Images LED sanded/png/20241001_151241 - Kopie2.png', cv2.IMREAD_GRAYSCALE) 

# Ensure the image is a NumPy array and of the correct data type
# image = np.array(image)
# Convert to uint8 if it's not already
image = image.astype(np.uint8)

# Apply Gaussian Blur to reduce noise and detail
blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Apply image thresholding to isolate features
_, thresh = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY_INV)

# Detect contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if option_1:
    # Prepare to store the center of the "+" sign
    center_of_plus = None

    # Loop through contours to identify "+" shape
    for cnt in contours:
        # Approximate the contour to remove noise
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        
        # Look for contour that might be the "+" by filtering based on area or shape
        area = cv2.contourArea(cnt)
        if 500 < area < 1000:  # Using area range to filter contours
            # Compute the center of the contour
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_of_plus = (cX, cY)
                break

    # Plotting the result
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    if center_of_plus:
        plt.scatter(center_of_plus[0], center_of_plus[1], color='red', s=100, label="Center of +")
        plt.legend()
    plt.show()

    center_of_plus

if option_2:
    # Filter based on area or aspect ratio (for a cross, width and height should be roughly equal)
    for cnt in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter based on aspect ratio (close to 1 for a cross) and area
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)
        
        if 0.8 < aspect_ratio < 1.2 and 500 < area < 1500:
            # Draw the contour and bounding box (for debugging)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            center_of_plus = (x + w // 2, y + h // 2)
            break

    # Display the result
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    if center_of_plus:
        plt.scatter(center_of_plus[0], center_of_plus[1], color='red', s=100, label="Center of +")
        plt.legend()
    plt.show()

if option_3:
    # Apply Gaussian Blur to reduce noise and detail
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Use Hough Line Transform to find lines in the image
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=70, maxLineGap=2)
    
    if lines is not None:
        # Prepare to store detected line segments
        vertical_lines = []
        horizontal_lines = []
        
        # Loop through all detected lines and categorize them
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle of the line
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Classify the line as horizontal or vertical
                if -10 <= angle <= 10:  # Almost horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
                elif 80 <= abs(angle) <= 100:  # Almost vertical
                    vertical_lines.append((x1, y1, x2, y2))

        # Look for a combination of horizontal and vertical lines that form a cross
        cross_center = None
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                # Check if the lines intersect or are close to each other (i.e., form a cross)
                hx1, hy1, hx2, hy2 = h_line
                vx1, vy1, vx2, vy2 = v_line
                
                # Check if the horizontal and vertical lines intersect
                if vx1 <= hx1 <= vx2 and hy1 <= vy1 <= hy2:
                    # This is a cross intersection, calculate the center
                    cross_center = (hx1, vy1)
                    break
            if cross_center:
                break

        # Draw detected lines (for visualization/debugging)
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for x1, y1, x2, y2 in vertical_lines:
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Mark the cross center, if found
        if cross_center:
            cv2.circle(output_image, cross_center, 10, (0, 0, 255), -1)
        
        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(output_image)
        plt.show()

        if cross_center:
            print(f"Detected cross at: {cross_center}")
        else:
            print("No cross detected.")