

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
    def __init__(self, img, ctr, savepath):
        self.img = img
        self.ctr = ctr # Reference Coordinates to Perspective Transform
        self.savepath = savepath

    # transform warped rectangle to straight rectangle ----------------------------------------------
    def transform_pixel_coordinates(self):
        height, width = image.shape[:2] # determine size of image

        # Calculate new coordinates to correct for distortion
        a1 = self.ctr[0][0] + ((self.ctr[0][1] - self.ctr[1][1])) / math.sin(math.atan(((self.ctr[0][1] - self.ctr[1][1])) / (self.ctr[1][0] - self.ctr[0][0])))
        a2 = self.ctr[0][1] + ((self.ctr[0][0] - self.ctr[3][0])) / math.sin(math.atan(((self.ctr[0][0] - self.ctr[3][0])) / (self.ctr[3][1] - self.ctr[0][1])))
        pts2 = np.float32([self.ctr[0], [a1, self.ctr[0][1]], [a1, a2], [self.ctr[0][0], a2]]) # corrected coordinates
        # Transform Perspective
        M = cv2.getPerspectiveTransform(self.ctr, pts2)
        transformed_image = cv2.warpPerspective(self.img, M, (width, height))
        print(transformed_image)
        print(type(transformed_image))

        # Save the transformed_image to an HDF5 file
        with h5py.File(self.savepath, 'w') as h5_file:
            h5_file.create_dataset('transformed', data=transformed_image)

        return transformed_image

#%% RUN CODE

# PARAMETER
# image from .h5 file
imagename = "20241004_094054_step_0_batch_0"
h5_file_path = 'G:/Limit/Lina Scholz/robot_files_20241004/20241004_094054_step_0_batch_0.h5'
with h5py.File(h5_file_path, 'r') as h5_file:
    # Angenommen, das Bild liegt in einem Dataset namens 'image'
    image = np.array(h5_file['image'])
# center
center = np.float32([[586, 758], [4802, 734], [4864, 2918], [508, 2936]])  
# path to save transformed image 
save_path = f'G:/Limit/Lina Scholz/robot_files_20241004/transformed/{imagename}_transformed.h5'
if not os.path.exists('G:/Limit/Lina Scholz/robot_files_20241004/transformed'):
    os.makedirs('G:/Limit/Lina Scholz/robot_files_20241004/transformed')

# EXECUTE
obj = TRANSFORM(image, center, save_path)
transformed = obj.transform_pixel_coordinates() # transform pixel coordinates to get "rectangular shape"


# CHECK

# Plot Input- und transformed Output Image
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(image, cmap='gray') # plot input image
axs[0].set_title("Input Image")
axs[1].imshow(transformed, cmap='gray') # plot output image
axs[1].set_title("Transformed Output Image")
# Add rectangle to check transformation
axs[1].add_patch(Rectangle((center[0][0] - .5, center[0][1] - .5), (center[1][0] - center[0][0]), (center[3][1] - center[0][1]), facecolor="none", ec='r', lw=1))
axs[0].add_patch(Rectangle((center[0][0] - .5, center[0][1] - .5), (center[1][0] - center[0][0]), (center[3][1] - center[0][1]), facecolor="none", ec='r', lw=1))
plt.show()





