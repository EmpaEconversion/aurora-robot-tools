"""Tests for the camera daemon using fake cameras."""

import socket
import sqlite3
import threading
import time
import tkinter as tk
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import zxingcpp

from aurora_robot_tools import config
from aurora_robot_tools.camera import camera_daemon as daemon
from aurora_robot_tools.camera import viewer
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
            "`Cathode Rack Position` INT, `Barcode` TEXT, `Last Completed Step` INT, `Current press number` INT, "
            "`Error Code` INT)",
        )
        cursor.execute(
            "INSERT INTO Cell_Assembly_Table VALUES "
            "(2, 5, 7, 9, NULL, 20, 1, 0), (3, 6, 8, 10, NULL, 30, 2, 0), (0, 4, 0, 0, NULL, 0, 0, 0)",
        )
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


class TestRunSummary:
    """Run status shown at the top of the camera window."""

    @pytest.mark.usefixtures("robot_db")
    def test_summary(self) -> None:
        """Run ID, number of cells and the latest step with its description."""
        summary = viewer.read_run_summary(config.DATABASE_FILEPATH)
        assert summary.error is None
        assert summary.run_id == "RUN1"
        assert summary.cells == 2
        assert summary.current_step == "Cell 2, current step: Place anode face up"

    @pytest.mark.usefixtures("robot_db")
    def test_latest_step(self) -> None:
        """The most recent timestamp is the current step."""
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            conn.execute("INSERT INTO Timestamp_Table VALUES (3, 80, 1, 9)")
        summary = viewer.read_run_summary(config.DATABASE_FILEPATH)
        assert summary.current_step == "Cell 3, current step: Place anode face down"

    @pytest.mark.usefixtures("robot_db")
    def test_run_complete(self) -> None:
        """Complete once every cell without an error has reached the last step."""
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            conn.execute("UPDATE Cell_Assembly_Table SET `Last Completed Step` = ? WHERE `Cell Number` = 2", (140,))
        assert viewer.read_run_summary(config.DATABASE_FILEPATH).current_step.startswith("Cell 2")
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            conn.execute("UPDATE Cell_Assembly_Table SET `Error Code` = 301 WHERE `Cell Number` = 3")
        assert viewer.read_run_summary(config.DATABASE_FILEPATH).current_step == "Run complete"

    @pytest.mark.usefixtures("robot_db")
    def test_not_complete_without_cells(self) -> None:
        """A run with no cells assigned is not complete."""
        with sqlite3.connect(config.DATABASE_FILEPATH) as conn:
            conn.execute("UPDATE Cell_Assembly_Table SET `Cell Number` = 0")
        assert viewer.read_run_summary(config.DATABASE_FILEPATH).current_step != "Run complete"

    def test_missing_database(self) -> None:
        """A missing database is reported, not created."""
        summary = viewer.read_run_summary(config.DATABASE_FILEPATH)
        assert summary.error is not None
        assert not config.DATABASE_FILEPATH.exists()


@pytest.fixture
def other_connection() -> Iterator[sqlite3.Connection]:
    """Second connection to the database, like an SQLite viewer, rolled back after the test."""
    conn = sqlite3.connect(config.DATABASE_FILEPATH, isolation_level=None)
    yield conn
    if conn.in_transaction:
        conn.execute("ROLLBACK")
    conn.close()


@pytest.mark.usefixtures("robot_db")
class TestDatabaseLock:
    """Viewer keeps working and warns while someone else holds a database lock."""

    def test_write_lock_detected(self, other_connection: sqlite3.Connection) -> None:
        """Uncommitted changes block the robot's writes but not reading the summary."""
        assert not viewer.database_write_locked(config.DATABASE_FILEPATH)
        other_connection.execute("BEGIN IMMEDIATE")
        other_connection.execute("UPDATE Settings_Table SET `value` = 'EDITED'")
        assert viewer.database_write_locked(config.DATABASE_FILEPATH)
        assert viewer.read_run_summary(config.DATABASE_FILEPATH).run_id == "RUN1"
        other_connection.execute("ROLLBACK")
        assert not viewer.database_write_locked(config.DATABASE_FILEPATH)

    def test_warning_after_lock_persists(self, other_connection: sqlite3.Connection) -> None:
        """Short locks are ignored, a lock that lasts is reported until it is released."""
        poller = viewer.SummaryPoller(config.DATABASE_FILEPATH, lock_warning_s=0.3)
        other_connection.execute("BEGIN IMMEDIATE")
        poller.poll()
        assert not poller.locked
        time.sleep(0.4)
        poller.poll()
        assert poller.locked
        other_connection.execute("ROLLBACK")
        poller.poll()
        assert not poller.locked

    def test_read_blocked_keeps_last_summary(self, other_connection: sqlite3.Connection) -> None:
        """While even reading is blocked, the last summary stays on screen."""
        poller = viewer.SummaryPoller(config.DATABASE_FILEPATH, lock_warning_s=0)
        poller.poll()
        other_connection.execute("BEGIN EXCLUSIVE")
        other_connection.execute("UPDATE Settings_Table SET `value` = 'EDITED'")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            viewer.read_run_summary(config.DATABASE_FILEPATH)
        poller.poll()
        assert poller.locked
        assert poller.summary.run_id == "RUN1"
        assert poller.summary.error is None

    def test_missing_database_is_not_locked(self, tmp_path: Path) -> None:
        """A missing database is not reported as locked, or created."""
        assert not viewer.database_write_locked(tmp_path / "missing.db")
        assert not (tmp_path / "missing.db").exists()

    def test_banner(self, cameras: list[Camera]) -> None:
        """Banner appears while locked and disappears when released."""
        stations = {"arm": daemon.Station(start(cameras, FakeCamera("arm", fps=30)))}
        try:
            window = viewer.CameraViewer(stations)
        except tk.TclError:
            pytest.skip("No display available for Tk")
        try:
            window.poller.locked = True
            window.refresh()
            assert window.lock_banner.winfo_manager() == "pack"
            window.poller.locked = False
            window.refresh()
            assert window.lock_banner.winfo_manager() == ""
        finally:
            window.root.destroy()


@pytest.mark.usefixtures("robot_db")
def test_socket_listener_survives_failed_capture(cameras: list[Camera], monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that fails on a locked database does not stop later commands."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    monkeypatch.setattr(config, "CAMERA_PORT", free_port)
    stations = {"arm": daemon.Station(start(cameras, FakeCamera("arm", fps=30)))}
    threading.Thread(target=daemon.socket_listener, args=(stations,), daemon=True).start()

    def send(command: str) -> bytes:
        for _ in range(50):
            try:
                client = socket.create_connection(("127.0.0.1", free_port), timeout=10)
                break
            except ConnectionRefusedError:
                time.sleep(0.05)
        with client:
            client.sendall(command.encode())
            return client.recv(1024)

    real_get_run_id = daemon.get_run_id
    calls = []

    def locked_once(cursor: sqlite3.Cursor) -> str:
        calls.append(1)
        if len(calls) == 1:
            msg = "database is locked"
            raise sqlite3.OperationalError(msg)
        return real_get_run_id(cursor)

    monkeypatch.setattr(daemon, "get_run_id", locked_once)
    assert send("capturearm") == b"0"  # Replies before reading the database, then fails
    assert send("capturearm") == b"0"
    deadline = time.monotonic() + 10
    while not images("arm_camera") and time.monotonic() < deadline:
        time.sleep(0.1)
    assert images("arm_camera") == ["cell_2_step_30.jpg"]


@pytest.mark.usefixtures("robot_db")
def test_viewer(cameras: list[Camera]) -> None:
    """Window shows every camera, the run summary and activity, then closes after confirming."""
    stations = daemon.build_stations(fake=True)
    for station in stations.values():
        start(cameras, station.camera)
    try:
        window = viewer.CameraViewer(stations)
    except tk.TclError:
        pytest.skip("No display available for Tk")
    seen: dict[str, object] = {}

    def check_and_close() -> None:
        seen["run"] = window.run_label.cget("text")
        seen["step"] = window.step_label.cget("text")
        seen["tiles"] = [tile._photo is not None for tile in window.tiles]  # noqa: SLF001
        seen["activity"] = window.activity.get("1.0", "end")
        window.confirm_quit()

    window.root.after(500, lambda: daemon.logger.info("Test capture message"))
    window.root.after(2000, check_and_close)
    with patch.object(viewer.messagebox, "askokcancel", return_value=True):
        window.run()

    assert seen["run"] == "Run: RUN1    Cells: 2"
    assert seen["step"] == "Cell 2, current step: Place anode face up"
    assert seen["tiles"] == [True, True, True]
    assert "Test capture message" in seen["activity"]


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
