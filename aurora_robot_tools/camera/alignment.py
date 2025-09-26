"""Functions for finding alignment from images.

Assumes that images are centred, i.e. the ideal alignment is at the centre of the image.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

step_radius = {
    0: 2.0,
    1: 10.0,
    2: 10.0,
    30: 7.5,
    40: 7.0,
    60: 8.0,
    100: 8.0,
    120: 9.0,
}
mm_to_px = 1600 / 20


def detect_circle(image: np.ndarray, step_radius_px: float) -> tuple:
    """Detect circle in image, try different color channels if one does not work."""

    def find_circle(img):
        """Detect the circle in the image using HoughCircles."""
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            1000,
            param1=50,
            param2=30,
            minRadius=int(step_radius_px * 0.95),
            maxRadius=int(step_radius_px * 1.05),
        )
        if circles is not None and len(circles) == 1:
            circle = circles[0][0]
            c_x_px = int(circle[0])
            c_y_px = int(circle[1])
            return c_x_px, c_y_px
        return None, None

    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)

    # Try green - red
    transformed_image = np.clip(g - r, 0, 255).astype(np.uint8)
    dx_px, dy_px = find_circle(transformed_image)
    if dx_px is not None:
        return dx_px, dy_px

    # If that fails, try blue - (green + red)
    transformed_image = np.clip(b - 0.5 * g - 0.5 * r, 0, 255).astype(np.uint8)
    return find_circle(transformed_image)


def process_folder(
    folder_path: Path | str,
) -> None:
    """Process all images in a folder."""
    folder_path = Path(folder_path)
    # Get all images with format cell_*_step_*.png
    # May also be in format cell_*_rack_*_step_*.png
    images = [
        *folder_path.glob("cell_*_step_*.png"),
        *folder_path.glob("cell_*_step_*.jpg"),
        *folder_path.glob("cell_*_step_*.jpeg"),
    ]
    if len(images) == 0:
        msg = f"No images found in {folder_path}."
        raise ValueError(msg)
    alignments = []
    not_found = []
    for image_path in tqdm(images, desc="Processing images"):
        # get cell number and step number from filename
        parts = image_path.stem.split("_")
        if len(parts) == 4:
            cell = int(parts[1])
            rack = cell
            step = int(parts[3])
        elif len(parts) == 6:
            cell = int(parts[1])
            rack = int(parts[3])
            step = int(parts[5])
        else:
            print(f"File {image_path.stem} not understood")
            continue
        # read image
        image = cv2.imread(str(image_path))
        # detect circle
        step_radius_px = step_radius.get(step, 10.0) * mm_to_px
        dx_px, dy_px = detect_circle(image, step_radius_px)
        if dx_px is None or dy_px is None:
            not_found.append(image_path)
            continue
        y, x, _ = image.shape
        dx_mm = (x // 2 - dx_px) / mm_to_px
        dy_mm = (y // 2 - dy_px) / mm_to_px
        alignments.append(
            {
                "Cell Number": cell,
                "Rack Position": rack,
                "Step Number": step,
                "dx_mm": dx_mm,
                "dy_mm": dy_mm,
            },
        )
        # draw circle on image and save to another file
        image = cv2.circle(image, (dx_px, dy_px), int(step_radius_px), (0, 0, 255), 2)
        image = cv2.line(image, (x // 2, 0), (x // 2, y), (0, 0, 0), 2)
        image = cv2.line(image, (0, y // 2), (x, y // 2), (0, 0, 0), 2)
        image = cv2.line(image, (dx_px, dy_px - 10), (dx_px, dy_px + 10), (0, 0, 255), 2)
        image = cv2.line(image, (dx_px - 10, dy_px), (dx_px + 10, dy_px), (0, 0, 255), 2)
        # save image with new name
        new_image_path = folder_path / f"detected/{image_path.stem}_detected.jpg"
        new_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(new_image_path), image)
    if len(not_found) > 0:
        print(f"Images with no circle found: {len(not_found)}")
        for image_path in not_found:
            print(image_path)

    with (folder_path / "alignment.json").open("w") as f:
        json.dump(
            alignments,
            f,
            indent=4,
        )
