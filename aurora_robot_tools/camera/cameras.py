"""Cameras that grab frames in a background thread, keeping the latest frame and a small preview."""

import contextlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from functools import cache
from typing import NamedTuple

import cv2
import gxipy as gx
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

logger = logging.getLogger(__name__)


class Preview(NamedTuple):
    """Downscaled copy of a frame, scale is preview size / full size."""

    frame_id: int
    image: np.ndarray
    scale: float


class Camera(ABC):
    """Base class for a camera.

    Reads frames in its own thread.
    Frames are stored read-only and shared with callers without copying.
    If the camera stops responding it is closed and reopened.
    """

    max_read_failures = 10
    reconnect_delay_s = 3.0

    def __init__(self, name: str, preview_width: int = 640) -> None:
        """Set up the camera, call start() to begin grabbing frames."""
        self.name = name
        self.preview_width = preview_width
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._preview: Preview | None = None
        self._frame_id = 0
        self._fps = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @abstractmethod
    def _open(self) -> None:
        """Open the device, raise if it is not available."""

    @abstractmethod
    def _read(self) -> np.ndarray | None:
        """Read one frame, None if no frame was received."""

    @abstractmethod
    def _close(self) -> None:
        """Release the device."""

    def start(self) -> None:
        """Start grabbing frames in a background thread."""
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-camera", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the grabber thread and release the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def connected(self) -> bool:
        """Whether the camera is open and producing frames."""
        return self._connected

    @property
    def status(self) -> str:
        """Short human-readable status."""
        return f"{self._fps:.1f} fps" if self._connected else "disconnected"

    def latest(self) -> np.ndarray | None:
        """Most recent full-size frame (read-only), None if no frame yet."""
        with self._cond:
            return self._frame

    def wait_for_new_frame(self, timeout: float) -> np.ndarray | None:
        """Wait for a frame captured after this call, None on timeout."""
        with self._cond:
            start_id = self._frame_id
            if self._cond.wait_for(lambda: self._frame_id > start_id, timeout):
                return self._frame
            return None

    def preview(self) -> Preview | None:
        """Writable copy of the latest preview, None if no frame yet."""
        with self._cond:
            preview = self._preview
        if preview is None:
            return None
        return preview._replace(image=preview.image.copy())

    def _run(self) -> None:
        warned = False
        while not self._stop.is_set():
            try:
                self._open()
            except Exception:  # noqa: BLE001
                level = logging.DEBUG if warned else logging.WARNING
                logger.log(level, "%s camera not available, retrying every %.0f s", self.name, self.reconnect_delay_s)
                logger.debug("Exception details:", exc_info=True)
                warned = True
                self._stop.wait(self.reconnect_delay_s)
                continue
            warned = False
            logger.info("%s camera connected", self.name)
            try:
                self._grab_until_failure()
            finally:
                self._connected = False
                with contextlib.suppress(Exception):
                    self._close()
            if not self._stop.is_set():
                logger.warning("%s camera stopped responding, reconnecting", self.name)

    def _grab_until_failure(self) -> None:
        failures = 0
        last_time = time.perf_counter()
        while not self._stop.is_set():
            try:
                frame = self._read()
            except Exception:  # noqa: BLE001
                logger.debug("%s camera read failed", self.name, exc_info=True)
                frame = None
            if frame is None or frame.size == 0:
                failures += 1
                if failures >= self.max_read_failures:
                    return
                self._stop.wait(0.1)
                continue
            failures = 0
            frame.flags.writeable = False
            preview = self._make_preview(frame)
            now = time.perf_counter()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            with self._cond:
                self._frame = frame
                self._frame_id += 1
                self._preview = Preview(self._frame_id, preview, preview.shape[1] / frame.shape[1])
                self._fps = fps if not self._connected else 0.8 * self._fps + 0.2 * fps
                self._connected = True
                self._cond.notify_all()

    def _make_preview(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if width <= self.preview_width:
            return frame
        size = (self.preview_width, round(height * self.preview_width / width))
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def find_usb_camera(query: str) -> int:
    """DirectShow index of the one USB camera whose name, VID:PID or path contains query."""
    cameras = enumerate_cameras(cv2.CAP_DSHOW)
    matches = [c for c in cameras if query.lower() in _describe_usb(c).lower()]
    if len(matches) != 1:
        found = ", ".join(_describe_usb(c) for c in cameras) or "none"
        msg = f"Expected one USB camera matching {query!r}, found {len(matches)}. Cameras: {found}"
        raise LookupError(msg)
    return matches[0].index


def _describe_usb(camera_info) -> str:  # noqa: ANN001
    vid_pid = f"{camera_info.vid:04X}:{camera_info.pid:04X}" if camera_info.vid is not None else "????:????"
    return f"{camera_info.name} ({vid_pid}) {camera_info.path}"


def _configure_capture(cap: cv2.VideoCapture, resolution: tuple[int, int] | None, fourcc: str | None) -> None:
    """Request a resolution (None for the largest) and optionally a pixel format."""
    width, height = resolution or (10000, 10000)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fourcc:  # With DirectShow this only takes effect after the resolution is set
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))


_usb_open_lock = threading.Lock()


class UsbCamera(Camera):
    """USB webcam opened with DirectShow.

    device is a DirectShow index, or part of the name, VID:PID or path shown by `aurora-rt listcams`.
    resolution None requests the largest the camera supports.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        device: int | str,
        resolution: tuple[int, int] | None = None,
        focus: int | None = None,
        fourcc: str | None = None,
        preview_width: int = 640,
    ) -> None:
        """Set up the camera, call start() to begin grabbing frames."""
        super().__init__(name, preview_width)
        self.device = device
        self.resolution = resolution
        self.focus = focus
        self.fourcc = fourcc
        self._cap: cv2.VideoCapture | None = None

    def _open(self) -> None:
        index = self.device if isinstance(self.device, int) else find_usb_camera(self.device)
        with _usb_open_lock:  # Each camera reaches its final format before the next one opens
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                msg = f"Could not open USB camera {self.device!r} (index {index})"
                raise OSError(msg)
            _configure_capture(cap, self.resolution, self.fourcc)
            cap.read()
            if self.focus is not None:
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, self.focus)
        self._cap = cap

    def _read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


@cache
def _gx_device_manager() -> gx.DeviceManager:
    return gx.DeviceManager()


_gx_open_lock = threading.Lock()


class GxiCamera(Camera):
    """Daheng gxipy camera in MONO8 with auto exposure.

    index starts at 1. frame_rate caps the stream to lower the work on the PC, None for the camera maximum.
    """

    def __init__(
        self,
        name: str,
        index: int = 1,
        frame_rate: float | None = 3.0,
        preview_width: int = 640,
        timeout_ms: int = 2000,
    ) -> None:
        """Set up the camera, call start() to begin grabbing frames."""
        super().__init__(name, preview_width)
        self.index = index
        self.frame_rate = frame_rate
        self.timeout_ms = timeout_ms
        self._device = None

    def _open(self) -> None:
        with _gx_open_lock:
            manager = _gx_device_manager()
            manager.update_device_list()
            device = manager.open_device_by_index(self.index)
        if device is None:
            msg = f"Could not open gxipy camera {self.index}"
            raise OSError(msg)
        try:
            device.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
            device.AcquisitionMode.set(gx.GxAcquisitionModeEntry.CONTINUOUS)
            device.ExposureAuto.set(gx.GxAutoEntry.CONTINUOUS)
            if self.frame_rate is not None:
                self._limit_frame_rate(device)
            device.stream_on()
        except Exception:
            device.close_device()
            raise
        self._device = device

    def _limit_frame_rate(self, device) -> None:  # noqa: ANN001
        if not device.AcquisitionFrameRateMode.is_implemented():
            logger.warning("%s camera does not support limiting the frame rate", self.name)
            return
        device.AcquisitionFrameRateMode.set(gx.GxSwitchEntry.ON)
        device.AcquisitionFrameRate.set(self.frame_rate)

    def _read(self) -> np.ndarray | None:
        image = self._device.data_stream[0].get_image(timeout=self.timeout_ms)
        return None if image is None else image.get_numpy_array()

    def _close(self) -> None:
        if self._device is not None:
            with contextlib.suppress(Exception):
                self._device.stream_off()
            self._device.close_device()
            self._device = None


class FakeCamera(Camera):
    """Synthetic camera for testing without hardware.

    Shows `image` if given, otherwise a grey gradient, with the name and frame number drawn top-left.
    Set `available = False` to simulate unplugging the camera.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        resolution: tuple[int, int] = (1280, 720),
        fps: float = 30.0,
        mono: bool = False,
        image: np.ndarray | None = None,
        preview_width: int = 640,
    ) -> None:
        """Set up the camera, call start() to begin grabbing frames."""
        super().__init__(name, preview_width)
        self.resolution = resolution
        self.fps = fps
        self.mono = mono
        self.image = image
        self.available = True
        self._base: np.ndarray | None = None
        self._count = 0
        self._next_time = 0.0

    def _open(self) -> None:
        if not self.available:
            msg = f"Fake {self.name} camera is unplugged"
            raise OSError(msg)
        if self.image is not None:
            self._base = self.image
        else:
            width, height = self.resolution
            self._base = np.tile(np.linspace(40, 200, width, dtype=np.uint8), (height, 1))
            if not self.mono:
                self._base = cv2.cvtColor(self._base, cv2.COLOR_GRAY2BGR)
        self._count = 0
        self._next_time = time.perf_counter()

    def _read(self) -> np.ndarray | None:
        self._next_time += 1 / self.fps
        delay = self._next_time - time.perf_counter()
        if delay > 0:
            self._stop.wait(delay)
        else:
            self._next_time = time.perf_counter()
        if not self.available:
            return None
        self._count += 1
        frame = self._base.copy()
        scale = frame.shape[1] / 640
        cv2.putText(
            frame,
            f"{self.name} #{self._count}",
            (int(10 * scale), int(30 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            (255, 255, 255),
            max(1, int(2 * scale)),
        )
        return frame

    def _close(self) -> None:
        self._base = None


def _frame_size(cap: cv2.VideoCapture) -> str:
    return f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"


def _probe_usb(index: int, backend: int, fourcc: str | None) -> str:
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            return "could not open"
        _configure_capture(cap, None, fourcc)
        ok, _ = cap.read()
        return f"opened at {_frame_size(cap)}, read {'ok' if ok else 'FAILED'}"
    finally:
        cap.release()


def _probe_usb_together(
    indices: tuple[int, ...],
    backend: int,
    resolution: tuple[int, int] | None,
    fourcc: str | None,
) -> str:
    """Open cameras one after another, each fully set up and read before opening the next."""
    caps = []
    results = []
    try:
        for index in indices:
            cap = cv2.VideoCapture(index, backend)
            caps.append(cap)
            if not cap.isOpened():
                results.append(f"[{index}] could not open")
                break
            _configure_capture(cap, resolution, fourcc)
            ok, _ = cap.read()
            results.append(f"[{index}] {_frame_size(cap)} read {'ok' if ok else 'FAILED'}")
        return ", ".join(results)
    finally:
        for cap in caps:
            cap.release()


def list_cameras() -> None:
    """Print connected cameras and test opening each USB camera."""
    print("Stop the camera daemon first, open cameras cannot be tested.\n")
    for backend_name, backend in (("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)):
        cameras = enumerate_cameras(backend)
        print(f"USB cameras ({backend_name}):")
        if not cameras:
            print("  none found")
        for camera_info in cameras:
            print(f"  [{camera_info.index}] {_describe_usb(camera_info)}")
            for fourcc in (None, "MJPG"):
                print(f"      {fourcc or 'default':7s}: {_probe_usb(camera_info.index, backend, fourcc)}")
        print()

    _print_usb_together_tests()
    _print_gx_cameras()


def _print_usb_together_tests() -> None:
    settings: list[tuple[str, tuple[int, int] | None, str | None]] = [
        ("max resolution, default format", None, None),
        ("max resolution, MJPG", None, "MJPG"),
        ("640x480, MJPG", (640, 480), "MJPG"),
    ]
    for backend_name, backend in (("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF)):
        indices = tuple(c.index for c in enumerate_cameras(backend))
        if len(indices) < 2:
            continue
        print(f"All USB cameras open together ({backend_name}), in both orders:")
        for label, resolution, fourcc in settings:
            for order in (indices, indices[::-1]):
                print(f"  {label:31s}: {_probe_usb_together(order, backend, resolution, fourcc)}", flush=True)
        print()


def _print_gx_cameras() -> None:
    print("gxipy cameras (index starts at 1):")
    try:
        num, infos = _gx_device_manager().update_device_list()
    except Exception as e:  # noqa: BLE001
        print(f"  could not list: {e!r}")
        return
    if not num:
        print("  none found")
    for i, info in enumerate(infos, start=1):
        print(f"  [{i}] {info.get('model_name', '?')}  serial {info.get('sn', '?')}")
