""" Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia

Simple script to capture an image from the camera and save as png and raw 12-bit numpy array.

"""

import os
from PIL import Image
import numpy as np
import gxipy as gx
from time import sleep
import h5py
import sqlite3

IMAGE_FOLDER = "C:/Aurora_images/"
DATABASE_FILEPATH = "C:/Modules/Database/chemspeedDB.db"

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
        if abs(diff) < 50:
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

# Stop data acquisition
cam.stream_off()

# Close connection
cam.close_device()

# Save last image
# Convert 12-bit image to 8-bit for saving as PNG
numpy_image_8bit = (numpy_image >> 4).astype(np.uint8)
im = Image.fromarray(numpy_image_8bit)

# Get Run ID from database and cell/press numbers from database
with sqlite3.connect(DATABASE_FILEPATH) as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT `value` FROM Settings_Table WHERE `key` = 'Base Sample ID'")
    run_id = cursor.fetchone()[0]
    cursor.execute(
        "SELECT `Current Press Number`, `Cell Number`, `Last Completed Step` "
        "FROM Cell_Assembly_Table "
        "WHERE `Current Press Number` > 0 AND `Error Code` = 0 "
        "ORDER BY `Current Press Number` ASC"
    )
    press_cell_steps = cursor.fetchall()

# Make filename from press/cell/step numbers
folderpath = os.path.join(IMAGE_FOLDER, run_id)
filename = "_".join([f"p{p}c{c}s{s}" for p,c,s in press_cell_steps])

# Make sure folder exists
if not os.path.exists(folderpath):
    os.makedirs(folderpath)

# Save lossy compressed png image, make sure filename doesn't already exist
if os.path.exists(os.path.join(folderpath, filename + ".png")):
    i = 1
    while os.path.exists(os.path.join(folderpath, filename + f"_{i}.png")):
        i += 1
    filename += f"_{i}"
im.save(os.path.join(folderpath, filename + ".png"), compress_level=9)

# Save lossless compressed raw 12-bit numpy array
if os.path.exists(os.path.join(folderpath, filename + ".h5")):
    i = 1
    while os.path.exists(os.path.join(folderpath, filename + f"_{i}.h5")):
        i += 1
    filename += f"_{i}"
with h5py.File(os.path.join(folderpath, filename + ".h5"), 'w') as f:
    f.create_dataset('image', data=numpy_image, compression='gzip', compression_opts=9)
