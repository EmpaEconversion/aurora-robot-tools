"""Daemon for live view of bottom-up camera, listens for capture command."""

import contextlib
import logging
import socket
import sqlite3
import threading
from collections.abc import Callable
from functools import partial
from time import sleep

import cv2
import gxipy as gx
import numpy as np
import zxingcpp

from aurora_robot_tools import config
from aurora_robot_tools.camera.ringlight import set_light

logger = logging.getLogger(__name__)

PHOTO_PATH = config.IMAGE_DIR
step_radius = {k: v.get("Radius", 10.0) for k, v in config.STEP_DEFINITION.items()}
mm_to_px = config.MM_TO_PX
radius_mm = 10.0
coords = (None, None)
last_frame_b = None
last_frame_t = None


def get_run_id(cursor: sqlite3.Cursor) -> str:
    """Get the base sample ID of the current run."""
    cursor.execute("SELECT `value` from Settings_Table WHERE `key` = 'Base Sample ID'")
    return cursor.fetchone()[0]


def get_current_cell_step(cursor: sqlite3.Cursor) -> tuple[int, int]:
    """Get the cell and step number of the latest incomplete step, (0, 0) if none."""
    cursor.execute(
        "SELECT `Cell Number`, `Step Number` from Timestamp_Table "
        "WHERE `Complete` = 0 ORDER BY `Timestamp` DESC LIMIT 1",
    )
    return cursor.fetchone() or (0, 0)


def save_photo(frame: np.ndarray, run_id: str, subdir: str, label: str) -> None:
    """Save a frame as a jpg in the run's image folder."""
    photo_path = PHOTO_PATH / run_id / subdir / f"{label}.jpg"
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(photo_path), frame)
    logger.info("Frame saved as %s", photo_path)


def get_frame(
    getter: Callable[[], np.ndarray | None],
    client_socket: socket.socket,
    camera_name: str,
) -> np.ndarray | None:
    """Get a copy of the latest frame, or reply failure to the client if there is none."""
    frame = getter()
    if frame is None:
        sleep(1)
        frame = getter()
    if frame is None:
        logger.info("No frame captured from %s camera", camera_name)
        client_socket.sendall(b"1")
        return None
    return frame.copy()


def capture_bottom(client_socket: socket.socket, read_qr: bool = False) -> None:
    """Capture an image from the bottom camera."""
    logger.info("Capturing from bottom camera")
    captured_frame = get_frame(lambda: last_frame_b, client_socket, "bottom")
    if captured_frame is None:
        return
    if not read_qr:  # If QR, wait until it is detected before moving on
        client_socket.sendall(b"0")
    # get current cell, press, step numbers from database
    with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        run_id = get_run_id(cursor)
        cell_number, step_number = get_current_cell_step(cursor)
        cursor.execute(
            "SELECT `Rack Position`, `Anode Rack Position`, `Cathode Rack Position` "  # noqa: S608
            "FROM Cell_Assembly_Table "
            f"WHERE `Cell Number` = {cell_number}",
        )
        result = cursor.fetchone()
        result = result if result else (0, 0, 0)
        if step_number in [30, 80]:  # Anode
            rack_position = result[1]
        elif step_number in [40, 90]:  # Cathode
            rack_position = result[2]
        elif step_number in [1, 2]:  # needle-head reference
            rack_position = 0
        elif step_number in [3, 4, 5, 6]:  # pressing tool
            rack_position = cell_number
        else:  # Other components
            rack_position = result[0]
        label = f"cell_{cell_number}_rack_{rack_position}_step_{step_number}"
    global radius_mm
    radius_mm = step_radius.get(int(step_number), 10.0)

    # If QR, try to read it and update db
    if read_qr:
        qr = detect_qr_code(captured_frame)
        i = 1
        attempts = 3
        while not qr and i < attempts:
            # Try again
            logger.info(f"Could not detect QR, attempt number {i + 1}")
            captured_frame = last_frame_b.copy()
            qr = detect_qr_code(captured_frame)
            i += 1
        if qr:
            logger.info("Found QR code: %s", qr)
            if cell_number:
                with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE Cell_Assembly_Table SET `Barcode` = ? WHERE `Cell Number` = ?",
                        (qr, cell_number),
                    )
                logger.info("Updated barcode in database")
            else:
                logger.info("No cell number, cannot update database")
        else:
            logger.info("Could not detect QR code")
        client_socket.sendall(b"0")  # Let autosuite continue

    # Detect circle in image
    global coords
    coords = detect_circle(captured_frame, radius_mm * mm_to_px)
    if coords[0] is not None:
        x = captured_frame.shape[1]
        y = captured_frame.shape[0]
        dx_mm = (x // 2 - coords[0]) / mm_to_px
        dy_mm = (y // 2 - coords[1]) / mm_to_px
        logger.info("Misalignment x: %d mm, y: %d mm", dx_mm, dy_mm)
        if result[0] > 0:
            write_coords_to_db(cell_number, step_number, rack_position, dx_mm, dy_mm)
    else:
        logger.info("Could not detect circle")
    save_photo(captured_frame, run_id, "bottom_camera", label)


def _read_single_qr(image: np.ndarray) -> str | None:
    """Return the text if the image contains exactly one text QR code."""
    results = zxingcpp.read_barcodes(image)
    if results and len(results) == 1 and results[0].format.name == "QRCode" and results[0].content_type.name == "Text":
        return results[0].text
    return None


QR_PREPROCESSING_STEPS: tuple[Callable[[np.ndarray], np.ndarray], ...] = (
    lambda img: img,
    lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
    cv2.equalizeHist,
    lambda img: cv2.GaussianBlur(img, (5, 5), 0),
)


def detect_qr_code(frame: np.ndarray) -> str | None:
    """Detect QR code from an image, applying each preprocessing step in turn until one is found."""
    for step in QR_PREPROCESSING_STEPS:
        frame = step(frame)
        qr = _read_single_qr(frame)
        if qr is not None:
            return qr
    return None


def capture_top(client_socket: socket.socket) -> None:
    """Capture an image from the top camera."""
    logger.info("Capturing from top camera")
    captured_frame = get_frame(lambda: last_frame_t, client_socket, "top")
    if captured_frame is None:
        return
    client_socket.sendall(b"0")
    # get current cell, press, step numbers from database
    with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        run_id = get_run_id(cursor)
        cursor.execute(
            "SELECT `Cell Number`, `Last Completed Step`, `Current press number` from Cell_Assembly_Table "
            "WHERE `Current press number` > 0 "
            "ORDER BY `Current press number` ASC",
        )
        results = cursor.fetchall()
        results = results if results else [(0, 0, 0)]
        label = "_".join([f"p{p}c{c}s{s}" for p, c, s in results])
    save_photo(captured_frame, run_id, "top_camera", label)


def write_coords_to_db(cell: int, step: int, rack: int, dx_mm: float, dy_mm: float) -> None:
    """Write the coordinates to the database."""
    with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        # insert Cell Number, Step Number, Rack Position dx_mm, dy_mm into Calibration_Table
        cursor.execute(
            "INSERT INTO Calibration_Table "
            "(`Cell Number`, `Step Number`, `Rack Position`, `dx_mm`, `dy_mm`) "
            "VALUES (?, ?, ?, ?, ?)",
            (cell, step, rack, dx_mm, dy_mm),
        )
        conn.commit()


def detect_circle(image: np.ndarray, step_radius_px: float) -> tuple:
    """Detect the circle in the image using HoughCircles."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        2,
        100,
        param1=50,
        param2=25,
        minRadius=int(step_radius_px * 0.98),
        maxRadius=int(step_radius_px * 1.02),
    )
    if circles is not None and len(circles) == 1:
        circle = circles[0][0]
        c_x_px = int(circle[0])
        c_y_px = int(circle[1])
        return c_x_px, c_y_px
    return None, None


def shrink_frame(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Shrink the frame by a ratio."""
    x = frame.shape[1]
    y = frame.shape[0]
    return cv2.resize(frame, [x // ratio, y // ratio])


def add_target(frame: np.ndarray, coords: tuple, radius_mm: float, ratio: float) -> np.ndarray:
    """Add target circles to the frame."""
    x = frame.shape[1]
    y = frame.shape[0]
    frame = cv2.circle(frame, (x // 2, y // 2), int(radius_mm * mm_to_px / ratio), (0, 0, 255))
    frame = cv2.line(frame, (x // 2, 0), (x // 2, y), (0, 0, 255))
    frame = cv2.line(frame, (0, y // 2), (x, y // 2), (0, 0, 255))
    if coords[0] is not None:
        resized_coords = coords[0] // ratio, coords[1] // ratio
        frame = cv2.circle(frame, resized_coords, int(radius_mm * mm_to_px / ratio), (0, 255, 0))
        frame = cv2.line(
            frame,
            (resized_coords[0], resized_coords[1] + 10),
            (resized_coords[0], resized_coords[1] - 10),
            (0, 255, 0),
        )
        frame = cv2.line(
            frame,
            (resized_coords[0] + 10, resized_coords[1]),
            (resized_coords[0] - 10, resized_coords[1]),
            (0, 255, 0),
        )
    return frame


COMMANDS: dict[str, Callable[[socket.socket], None]] = {
    "capturebottom": capture_bottom,
    "capturebottomqr": partial(capture_bottom, read_qr=True),
    "capturetop": capture_top,
}


def socket_listener() -> None:
    """Capture images when requested by socket connection."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", config.CAMERA_PORT))
    server_socket.listen(1)
    logger.info("Listening for connections...")
    while True:
        client_socket, addr = server_socket.accept()
        logger.info("Connection from %s", addr)
        data = client_socket.recv(1024).decode().strip()
        logger.info("Command: %s", data)
        handler = COMMANDS.get(data)
        if handler is None:
            logger.warning("Unknown command: %s", data)
        else:
            handler(client_socket)
        client_socket.close()


def main() -> None:
    """Start webcam, show in window, listen for capture command."""
    global last_frame_b
    global last_frame_t

    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", config.CAMERA_PORT))
        server_socket.close()
    except OSError:
        logger.critical("Cameras are already running!")
        logger.debug("Exception details:", exc_info=True)
        return

    try:
        set_light("party")
    except Exception:
        logger.warning("Lights not working, continuing without...")
        logger.debug("Exception details:", exc_info=True)

    thread = threading.Thread(target=socket_listener, daemon=True)
    thread.start()
    logger.info("Started listening")

    logger.critical("Starting cameras, press q to quit.")

    # Connect to first USB webcam with DirectShow
    try:
        logger.info("Loading bottom camera...")
        cam_b = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cam_b.set(3, 10000)  # Set max frame size
        cam_b.set(4, 10000)
        ret, frame = cam_b.read()
        cam_b.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cam_b.set(28, 1023)  # Set focus to closest distance
        ret, frame = cam_b.read()
    except Exception:
        cam_b = None
        logger.warning("Bottom camera not available")
        logger.debug("Exception details:", exc_info=True)

    # Connect to first gxipy camera
    try:
        logger.info("Loading top camera...")
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        cam_t = device_manager.open_device_by_index(1)
        if cam_t is not None:
            cam_t.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
            cam_t.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
            cam_t.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)
            cam_t.stream_on()
            frame_t = cam_t.data_stream[0].get_image().get_numpy_array()
            if not isinstance(frame_t, np.ndarray) or frame_t.shape[0] == 0 or frame_t.shape[1] == 0:
                raise ValueError("Couldn't get an image from topcam")
    except Exception:
        cam_t = None
        logger.warning("Top camera not available")
        logger.debug("Exception details:", exc_info=True)

    if cam_b is None and cam_t is None:
        logger.critical("No cameras available, exiting.")
        with contextlib.suppress(Exception):
            set_light("off")
        return

    # Set light to white to take photos
    try:
        set_light("b")
    except Exception:
        logger.warning("Lights not working, continuing without...")
        logger.debug("Exception details:", exc_info=True)

    logger.info("Ready to capture images.")
    try:
        while True:
            # Update bottom camera frame
            if cam_b is not None:
                ret, frame_b = cam_b.read()
                if isinstance(frame_b, np.ndarray):
                    last_frame_b = frame_b.copy()
                    frame_b = shrink_frame(frame_b, 4)
                    frame_b = add_target(frame_b, coords, radius_mm, 4)
                    cv2.imshow("Bottom camera", frame_b)

            # Update top camera frame
            if cam_t is not None:
                frame_t = cam_t.data_stream[0].get_image().get_numpy_array()
                if isinstance(frame_t, np.ndarray):
                    last_frame_t = frame_t.copy()
                    frame_t = shrink_frame(frame_t, 8)
                    cv2.imshow("Top camera", frame_t)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cam_t is not None:
            cam_t.stream_off()
            cam_t.close_device()
        if cam_b is not None:
            cam_b.release()
        with contextlib.suppress(Exception):
            set_light("off")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
