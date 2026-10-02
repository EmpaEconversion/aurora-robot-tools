"""Daemon for live view of the robot cameras, listens for capture commands."""

import contextlib
import logging
import socket
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import cv2
import numpy as np
import zxingcpp

from aurora_robot_tools import config
from aurora_robot_tools.camera.cameras import Camera, FakeCamera, GxiCamera, UsbCamera
from aurora_robot_tools.camera.ringlight import set_light

logger = logging.getLogger(__name__)

step_radius = {k: v.get("Radius", 10.0) for k, v in config.STEP_DEFINITION.items()}
STARTUP_TIMEOUT_S = 15.0


class CircleTarget:
    """Finds a circle with some radius and draws it on previews."""

    def __init__(self, mm_to_px: float, radius_mm: float = 10.0) -> None:
        """Set the pixel calibration and initial target radius."""
        self.mm_to_px = mm_to_px
        self._lock = threading.Lock()
        self._radius_mm = radius_mm
        self._coords: tuple[int, int] | None = None

    def locate(self, frame: np.ndarray, radius_mm: float) -> tuple[float, float] | None:
        """Detect the circle and return its offset from the image centre in mm, None if not found."""
        x, y = detect_circle(frame, radius_mm * self.mm_to_px)
        coords = None if x is None else (x, y)
        with self._lock:
            self._radius_mm = radius_mm
            self._coords = coords
        if coords is None:
            return None
        height, width = frame.shape[:2]
        return (width // 2 - x) / self.mm_to_px, (height // 2 - y) / self.mm_to_px

    def overlay(self, image: np.ndarray, scale: float) -> np.ndarray:
        """Draw the centre target and the last detected circle on a preview image."""
        with self._lock:
            radius_mm, coords = self._radius_mm, self._coords
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        height, width = image.shape[:2]
        radius_px = int(radius_mm * self.mm_to_px * scale)
        cv2.circle(image, (width // 2, height // 2), radius_px, (0, 0, 255))
        cv2.line(image, (width // 2, 0), (width // 2, height), (0, 0, 255))
        cv2.line(image, (0, height // 2), (width, height // 2), (0, 0, 255))
        if coords is not None:
            cx, cy = int(coords[0] * scale), int(coords[1] * scale)
            cv2.circle(image, (cx, cy), radius_px, (0, 255, 0))
            cv2.line(image, (cx, cy + 10), (cx, cy - 10), (0, 255, 0))
            cv2.line(image, (cx + 10, cy), (cx - 10, cy), (0, 255, 0))
        return image


@dataclass
class Station:
    """A camera with a job on the robot."""

    camera: Camera
    target: CircleTarget | None = None

    @property
    def name(self) -> str:
        """Name of the camera's role, e.g. bottom."""
        return self.camera.name


def fake_bottom_scene(
    resolution: tuple[int, int] = (2304, 1536),
    radius_mm: float = 7.5,
    offset_px: tuple[int, int] = (40, -25),
    background: int = 200,
    disc: int = 170,
) -> np.ndarray:
    """Grey image with slightly darker circle."""
    width, height = resolution
    image = np.full((height, width, 3), background, np.uint8)
    centre = (width // 2 + offset_px[0], height // 2 + offset_px[1])
    cv2.circle(image, centre, int(radius_mm * config.MM_TO_PX), (disc, disc, disc), -1)
    return image


def build_stations(fake: bool = False) -> dict[str, Station]:
    """Assign cameras to their roles on the robot, or fake cameras of similar size and speed."""
    if fake:
        bottom: Camera = FakeCamera("bottom", (2304, 1536), fps=2, image=fake_bottom_scene())
        top: Camera = FakeCamera("top", (5472, 3648), fps=3, mono=True)
        arm: Camera = FakeCamera("arm", (1280, 720), fps=30)
    else:
        bottom = UsbCamera("bottom", config.BOTTOM_CAMERA, focus=1023)
        top = GxiCamera("top", config.TOP_CAMERA_INDEX)
        arm = UsbCamera("arm", config.ARM_CAMERA, fourcc="MJPG")
    stations = [Station(bottom, CircleTarget(config.MM_TO_PX)), Station(top), Station(arm)]
    return {s.name: s for s in stations}


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
    photo_path = config.IMAGE_DIR / run_id / subdir / f"{label}.jpg"
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(photo_path), frame)
    logger.info("Frame saved as %s", photo_path)


def get_frame(station: Station, client_socket: socket.socket) -> np.ndarray | None:
    """Get the latest frame (read-only), or reply failure to the client if there is none."""
    frame = station.camera.latest()
    if frame is None:
        frame = station.camera.wait_for_new_frame(timeout=1)
    if frame is None:
        logger.info("No frame captured from %s camera", station.name)
        client_socket.sendall(b"1")
    return frame


def capture_bottom(station: Station, client_socket: socket.socket, read_qr: bool = False) -> None:
    """Capture an image from the bottom camera."""
    logger.info("Capturing from %s camera", station.name)
    captured_frame = get_frame(station, client_socket)
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
    radius_mm = step_radius.get(int(step_number), 10.0)

    # If QR, try to read it and update db
    if read_qr:
        qr = detect_qr_code(captured_frame)
        i = 1
        attempts = 3
        while not qr and i < attempts:
            logger.info("Could not detect QR, attempt number %d", i + 1)
            new_frame = station.camera.wait_for_new_frame(timeout=1)
            if new_frame is not None:
                captured_frame = new_frame
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

    if station.target is not None:
        offset = station.target.locate(captured_frame, radius_mm)
        if offset is not None:
            dx_mm, dy_mm = offset
            logger.info("Misalignment x: %.2f mm, y: %.2f mm", dx_mm, dy_mm)
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


def capture_top(station: Station, client_socket: socket.socket) -> None:
    """Capture an image from the top camera."""
    logger.info("Capturing from %s camera", station.name)
    captured_frame = get_frame(station, client_socket)
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


def capture_arm(station: Station, client_socket: socket.socket) -> None:
    """Capture an image from the arm camera."""
    logger.info("Capturing from %s camera", station.name)
    captured_frame = get_frame(station, client_socket)
    if captured_frame is None:
        return
    client_socket.sendall(b"0")
    with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        run_id = get_run_id(cursor)
        cell_number, step_number = get_current_cell_step(cursor)
    save_photo(captured_frame, run_id, "arm_camera", f"cell_{cell_number}_step_{step_number}")


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


# command: (station name, handler)
COMMANDS: dict[str, tuple[str, Callable[[Station, socket.socket], None]]] = {
    "capturebottom": ("bottom", capture_bottom),
    "capturebottomqr": ("bottom", partial(capture_bottom, read_qr=True)),
    "capturetop": ("top", capture_top),
    "capturearm": ("arm", capture_arm),
}


def socket_listener(stations: dict[str, Station]) -> None:
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
        station_name, handler = COMMANDS.get(data, (None, None))
        station = stations.get(station_name)
        if handler is None:
            logger.warning("Unknown command: %s", data)
        elif station is None:
            logger.warning("No %s camera configured for command %s", station_name, data)
            client_socket.sendall(b"1")
        else:
            handler(station, client_socket)
        client_socket.close()


def window_title(station: Station) -> str:
    """Title of the station's preview window."""
    return f"{station.name.capitalize()} camera"


def show_feeds(stations: dict[str, Station]) -> None:
    """Show each camera's preview in a window until q is pressed or a window is closed."""
    shown: dict[str, int] = {}
    next_title_update = 0.0
    while True:
        for station in stations.values():
            preview = station.camera.preview()
            if preview is None or shown.get(station.name) == preview.frame_id:
                continue
            shown[station.name] = preview.frame_id
            image = preview.image
            if station.target is not None:
                image = station.target.overlay(image, preview.scale)
            cv2.imshow(window_title(station), image)

        if time.monotonic() > next_title_update:
            next_title_update = time.monotonic() + 1
            for name in shown:
                station = stations[name]
                cv2.setWindowTitle(window_title(station), f"{window_title(station)} - {station.camera.status}")

        if cv2.waitKey(10) & 0xFF == ord("q"):
            return
        for name in shown:
            if cv2.getWindowProperty(window_title(stations[name]), cv2.WND_PROP_VISIBLE) < 1:
                return


def wait_for_any_camera(stations: dict[str, Station], timeout: float) -> bool:
    """Wait until at least one camera is producing frames."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(s.camera.connected for s in stations.values()):
            return True
        time.sleep(0.2)
    return False


def try_set_light(mode: str, fake: bool) -> None:
    """Set the ring light, logging instead of raising if it is not working."""
    if fake:
        logger.info("Fake cameras, not setting light to %s", mode)
        return
    try:
        set_light(mode)
    except Exception:  # noqa: BLE001
        logger.warning("Lights not working, continuing without...")
        logger.debug("Exception details:", exc_info=True)


def main(fake: bool = False) -> None:
    """Start cameras, show their feeds, listen for capture commands."""
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("127.0.0.1", config.CAMERA_PORT))
        server_socket.close()
    except OSError:
        logger.critical("Cameras are already running!")
        logger.debug("Exception details:", exc_info=True)
        return

    try_set_light("party", fake)
    if fake:
        logger.warning("Using fake cameras, capture commands still write to the database and image folder.")
    logger.critical("Starting cameras, press q or close a camera window to quit.")
    stations = build_stations(fake)
    for station in stations.values():
        station.camera.start()

    try:
        if not wait_for_any_camera(stations, STARTUP_TIMEOUT_S):
            logger.critical("No cameras available, exiting.")
            return

        thread = threading.Thread(target=socket_listener, args=(stations,), daemon=True)
        thread.start()
        logger.info("Started listening")

        try_set_light("b", fake)  # White light to take photos
        logger.info("Ready to capture images.")
        show_feeds(stations)
    finally:
        for station in stations.values():
            station.camera.stop()
        if not fake:
            with contextlib.suppress(Exception):
                set_light("off")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
