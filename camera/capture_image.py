import os
import sys
from datetime import datetime
from PIL import Image
import numpy as np
import gxipy as gx
from time import sleep
import h5py

# get argument from command line
if len(sys.argv) > 1:
    step = str(int(sys.argv[1]))
    if len(sys.argv) > 2:
        run_id = sys.argv[2]
    else:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
else:
    step = datetime.now().strftime('%H%M%S')

IMAGE_FOLDER = "C:/Aurora_images/"
PICKLE_FOLDER = "C:/Aurora_images/raw/"

# if folder doesn't exist, create it
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)
if not os.path.exists(PICKLE_FOLDER):
    os.makedirs(PICKLE_FOLDER)

# Connect to camera
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
print(f"Number of enumerated devices is {dev_num}")
if dev_num == 0:
    print("Number of enumerated devices is 0")
cam = device_manager.open_device_by_index(1)

# set pixel format to 12-bit
cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO12)

# set continuous trigger
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

# set continuous acquisition
cam.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)

# set auto exposure
cam.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)

# start data acquisition
cam.stream_on()

# grab images until exposure is stable
avg_brightness = 0
prev_avg_brightness = 0
stable = 0
failed = 0
for i in range(500):
    raw_image = cam.data_stream[0].get_image()
    if raw_image:
        numpy_image = raw_image.get_numpy_array()
        prev_avg_brightness = avg_brightness
        avg_brightness = np.mean(numpy_image)
        diff = avg_brightness - prev_avg_brightness
        if abs(diff) < 20:
            stable += 1
        else:
            stable = 0
        if stable > 20:
            break
    else:
        print("didn't get anything")
        failed += 1
        if failed >= 10:
            raise ValueError
        sleep(1)

# Save last image
# Convert 12-bit image to 8-bit for saving as PNG
numpy_image_8bit = (numpy_image >> 4).astype(np.uint8)
im = Image.fromarray(numpy_image_8bit)

# Check if filename already exists and add a number to it, save as png
base_path = os.path.join(IMAGE_FOLDER, run_id)
i = 1
filename = step
while os.path.exists(base_path, filename + ".png"):
    filename = step + "_" + str(i)
    i += 1
im.save(IMAGE_FOLDER + filename + ".png")

# Also save raw 12-bit numpy array in HDF5 format with compression
base_path = os.path.join(PICKLE_FOLDER, run_id)
i = 1
filename = step
while os.path.exists(base_path, filename + ".h5"):
    filename = step + "_" + str(i)
    i += 1
with h5py.File(PICKLE_FOLDER + filename + ".h5", 'w') as f:
    f.create_dataset('image', data=numpy_image, compression='gzip', compression_opts=9)

# Stop data acquisition
cam.stream_off()

# Close connection
cam.close_device()
