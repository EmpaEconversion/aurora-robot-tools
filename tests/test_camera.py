"""Tests for the camera daemon using fake cameras."""

import socket
import sqlite3
import threading
import time
from collections.abc import Iterator
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import zxingcpp

from aurora_robot_tools import config
from aurora_robot_tools.camera import camera_daemon as daemon
from aurora_robot_tools.camera.cameras import Camera, FakeCamera


class FakeSocket:
    """Records what the daemon sends back to the client."""

    def __init__(self) -> None:
        """Start with nothing sent."""
        self.sent: list[bytes] = []

    def sendall(self, data: bytes) -> None:
        """Record sent data."""
        self.sent.append(data)


@pytest.fixture
def cameras() -> Iterator[list[Camera]]:
    """Cameras appended to this list are stopped after the test."""
    started: list[Camera] = []
    yield started
    for camera in started:
        camera.stop()


def start(cameras: list[Camera], camera: Camera, timeout: float = 5) -> Camera:
    """Start a camera and wait for its first frame."""
    cameras.append(camera)
    camera.start()
    assert camera.wait_for_new_frame(timeout) is not None
    return camera


@pytest.fixture
def robot_db() -> None:
    """Database with the tables the camera daemon reads and writes."""
    with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE Settings_Table (`key` TEXT, `value` TEXT)")
        cursor.execute("INSERT INTO Settings_Table VALUES ('Base Sample ID', 'RUN1')")
        cursor.execute(
            "CREATE TABLE Timestamp_Table (`Cell Number` INT, `Step Number` INT, `Complete` INT, `Timestamp` INT)",
        )
        cursor.execute("INSERT INTO Timestamp_Table VALUES (1, 10, 1, 1), (2, 30, 0, 5), (3, 40, 0, 2)")
        cursor.execute(
            "CREATE TABLE Cell_Assembly_Table (`Cell Number` INT, `Rack Position` INT, `Anode Rack Position` INT, "
            "`Cathode Rack Position` INT, `Barcode` TEXT, `Last Completed Step` INT, `Current press number` INT)",
        )
        cursor.execute("INSERT INTO Cell_Assembly_Table VALUES (2, 5, 7, 9, NULL, 20, 1), (3, 6, 8, 10, NULL, 30, 2)")
        cursor.execute(
            "CREATE TABLE Calibration_Table (`Cell Number` INT, `Step Number` INT, `Rack Position` INT, "
            "`dx_mm` REAL, `dy_mm` REAL)",
        )


def qr_scene(text: str) -> np.ndarray:
    """Bottom camera scene with a QR code in the corner."""
    image = daemon.fake_bottom_scene()
    barcode = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.QRCode)
    qr = np.array(zxingcpp.write_barcode_to_image(barcode, scale=6))
    image[100 : 100 + qr.shape[0], 20 : 20 + qr.shape[1]] = qr[..., None]
    return image


def images(sub_dir: str) -> list[str]:
    """Names of photos saved for RUN1."""
    return sorted(p.name for p in (config.IMAGE_DIR / "RUN1" / sub_dir).glob("*.jpg"))


class TestCamera:
    """Behaviour shared by all cameras, using fakes."""

    def test_frames_are_read_only(self, cameras: list[Camera]) -> None:
        """Frames are shared with callers, so they must not be modified."""
        camera = start(cameras, FakeCamera("fake", fps=50))
        frame = camera.latest()
        assert frame.shape == (720, 1280, 3)
        assert not frame.flags.writeable

    def test_preview_is_downscaled_writable_copy(self, cameras: list[Camera]) -> None:
        """Previews are small and safe to draw on."""
        camera = start(cameras, FakeCamera("fake", (5472, 3648), fps=20, mono=True))
        preview = camera.preview()
        assert preview.image.shape == (427, 640)
        assert preview.scale == pytest.approx(640 / 5472)
        assert preview.image.flags.writeable
        assert camera.preview().image is not preview.image

    def test_wait_for_new_frame(self, cameras: list[Camera]) -> None:
        """Waiting gives a frame captured after the call, or None once stopped."""
        camera = start(cameras, FakeCamera("fake", fps=20))
        old = camera.latest()
        assert camera.wait_for_new_frame(1) is not old
        camera.stop()
        assert camera.wait_for_new_frame(0.3) is None

    def test_reconnects_after_unplug(self, cameras: list[Camera]) -> None:
        """A camera that stops responding is reopened when it comes back."""
        camera = FakeCamera("fake", fps=50)
        camera.reconnect_delay_s = 0.2
        start(cameras, camera)
        assert camera.connected

        camera.available = False
        deadline = time.monotonic() + 5
        while camera.connected and time.monotonic() < deadline:
            time.sleep(0.05)
        assert camera.status == "disconnected"

        camera.available = True
        assert camera.wait_for_new_frame(5) is not None
        assert camera.connected

    def test_slow_camera_does_not_slow_others(self, cameras: list[Camera]) -> None:
        """Each camera reads in its own thread."""
        fast = start(cameras, FakeCamera("fast", (640, 480), fps=50))
        start(cameras, FakeCamera("slow", (5472, 3648), fps=1, mono=True))
        first = fast.preview().frame_id
        time.sleep(1)
        assert fast.preview().frame_id - first > 25

    def test_stop_ends_thread(self, cameras: list[Camera]) -> None:
        """Stopping joins the grabber thread."""
        camera = start(cameras, FakeCamera("fake", fps=50))
        camera.stop()
        assert not camera._thread.is_alive()  # noqa: SLF001


class TestCircleTarget:
    """Alignment target used by the bottom camera."""

    def test_locate_offset_from_centre(self) -> None:
        """Offset is centre minus circle position, in mm."""
        target = daemon.CircleTarget(config.MM_TO_PX)
        offset = target.locate(daemon.fake_bottom_scene(offset_px=(40, -25)), 7.5)
        assert offset == pytest.approx((-40 / config.MM_TO_PX, 25 / config.MM_TO_PX), abs=0.05)

    def test_locate_none_without_circle(self) -> None:
        """No circle gives None."""
        target = daemon.CircleTarget(config.MM_TO_PX)
        assert target.locate(np.full((1536, 2304, 3), 200, np.uint8), 7.5) is None

    def test_overlay_on_mono_preview(self) -> None:
        """Overlay converts mono previews to colour so the target can be drawn in red."""
        target = daemon.CircleTarget(config.MM_TO_PX)
        image = target.overlay(np.zeros((427, 640), np.uint8), 0.1)
        assert image.shape == (427, 640, 3)
        assert image[213, :, 2].max() == 255


class TestCaptureHandlers:
    """Capture commands with fake cameras and a temporary database."""

    @pytest.mark.usefixtures("robot_db")
    def test_capture_bottom(self, cameras: list[Camera]) -> None:
        """Saves the photo with rack position and writes the misalignment."""
        camera = start(cameras, FakeCamera("bottom", (2304, 1536), fps=20, image=daemon.fake_bottom_scene()))
        station = daemon.Station(camera, daemon.CircleTarget(config.MM_TO_PX))
        client = FakeSocket()
        daemon.capture_bottom(station, client)
        assert client.sent == [b"0"]
        assert images("bottom_camera") == ["cell_2_rack_7_step_30.jpg"]
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            rows = conn.execute("SELECT * FROM Calibration_Table").fetchall()
        assert len(rows) == 1
        cell, step, rack, dx_mm, dy_mm = rows[0]
        assert (cell, step, rack) == (2, 30, 7)
        assert (dx_mm, dy_mm) == pytest.approx((-40 / config.MM_TO_PX, 25 / config.MM_TO_PX), abs=0.05)

    @pytest.mark.usefixtures("robot_db")
    def test_capture_bottom_qr(self, cameras: list[Camera]) -> None:
        """Reads the QR code into the database before replying."""
        camera = start(cameras, FakeCamera("bottom", (2304, 1536), fps=20, image=qr_scene("CELL-XYZ")))
        client = FakeSocket()
        daemon.capture_bottom(daemon.Station(camera, daemon.CircleTarget(config.MM_TO_PX)), client, read_qr=True)
        assert client.sent == [b"0"]
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            barcode = conn.execute("SELECT Barcode FROM Cell_Assembly_Table WHERE `Cell Number` = 2").fetchone()
        assert barcode == ("CELL-XYZ",)

    @pytest.mark.usefixtures("robot_db")
    def test_capture_top(self, cameras: list[Camera]) -> None:
        """Label lists every loaded press."""
        camera = start(cameras, FakeCamera("top", (5472, 3648), fps=5, mono=True))
        client = FakeSocket()
        daemon.capture_top(daemon.Station(camera), client)
        assert client.sent == [b"0"]
        assert images("top_camera") == ["p2c20s1_p3c30s2.jpg"]

    @pytest.mark.usefixtures("robot_db")
    def test_capture_arm(self, cameras: list[Camera]) -> None:
        """Label uses the current cell and step."""
        camera = start(cameras, FakeCamera("arm", fps=30))
        client = FakeSocket()
        daemon.capture_arm(daemon.Station(camera), client)
        assert client.sent == [b"0"]
        assert images("arm_camera") == ["cell_2_step_30.jpg"]

    def test_no_frame_replies_failure(self) -> None:
        """A camera that never produced a frame replies 1."""
        client = FakeSocket()
        daemon.capture_top(daemon.Station(FakeCamera("top")), client)
        assert client.sent == [b"1"]


@pytest.mark.usefixtures("robot_db")
def test_socket_listener(cameras: list[Camera], monkeypatch: pytest.MonkeyPatch) -> None:
    """Commands sent over the socket get the expected replies."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(config, "CAMERA_PORT", free_port)
    stations = {"arm": daemon.Station(start(cameras, FakeCamera("arm", fps=30)))}
    threading.Thread(target=daemon.socket_listener, args=(stations,), daemon=True).start()

    def send(command: str) -> bytes:
        for _ in range(50):
            try:
                client = socket.create_connection(("127.0.0.1", free_port), timeout=5)
                break
            except ConnectionRefusedError:
                time.sleep(0.05)
        with client:
            client.sendall(command.encode())
            return client.recv(1024)

    assert send("capturearm") == b"0"
    assert send("capturetop") == b"1"  # No top station
    assert send("nonsense") == b""
    assert images("arm_camera") == ["cell_2_step_30.jpg"]


def test_show_feeds(cameras: list[Camera]) -> None:
    """Every station is shown with its overlay until q is pressed."""
    stations = daemon.build_stations(fake=True)
    for station in stations.values():
        start(cameras, station.camera)
    shown: dict[str, np.ndarray] = {}
    deadline = time.monotonic() + 3

    def wait_key(_delay: int) -> int:
        return ord("q") if len(shown) == len(stations) or time.monotonic() > deadline else -1

    with (
        patch.object(cv2, "imshow", side_effect=shown.__setitem__),
        patch.object(cv2, "waitKey", side_effect=wait_key),
        patch.object(cv2, "setWindowTitle"),
        patch.object(cv2, "getWindowProperty", return_value=1.0),
    ):
        daemon.show_feeds(stations)

    assert set(shown) == {"Bottom camera", "Top camera", "Arm camera"}
    assert shown["Bottom camera"].shape[1] == 640
    assert shown["Bottom camera"][:, :, 2].max() == 255  # Red target overlay


def test_main_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full daemon runs with fake cameras without touching the ring light."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        monkeypatch.setattr(config, "CAMERA_PORT", s.getsockname()[1])
    with (
        patch.object(daemon, "set_light") as set_light,
        patch.object(daemon, "show_feeds") as show_feeds,
    ):
        daemon.main(fake=True)
    set_light.assert_not_called()
    stations = show_feeds.call_args.args[0]
    assert set(stations) == {"bottom", "top", "arm"}
    assert all(not s.camera._thread.is_alive() for s in stations.values())  # noqa: SLF001
