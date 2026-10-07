"""Tkinter window showing the camera feeds, the robot's current run and recent activity."""

import contextlib
import logging
import queue
import signal
import sqlite3
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from logging.handlers import QueueHandler
from pathlib import Path
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk

from aurora_robot_tools import config

if TYPE_CHECKING:
    from aurora_robot_tools.camera.camera_daemon import Station

logger = logging.getLogger(__name__)

ACTIVITY_LOGGER = "aurora_robot_tools.camera"
MAX_ACTIVITY_LINES = 500
FINAL_STEP = max(config.STEP_DEFINITION)


@dataclass
class RunSummary:
    """What the robot is doing, as display text."""

    run_id: str = "-"
    cells: int = 0
    current_step: str = "No steps yet"
    error: str | None = None


def is_lock_error(error: sqlite3.Error) -> bool:
    """Whether an SQLite error means another connection holds a lock."""
    return "locked" in str(error)


def database_write_locked(db_path: Path) -> bool:
    """Whether another connection holds the write lock, which also blocks the robot from writing.

    Takes the write lock without waiting and releases it straight away.
    """
    try:
        with contextlib.closing(
            sqlite3.connect(f"{db_path.as_uri()}?mode=rw", uri=True, timeout=0, isolation_level=None),
        ) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("ROLLBACK")
    except sqlite3.Error as e:
        return is_lock_error(e)
    return False


def read_run_summary(db_path: Path) -> RunSummary:
    """Read the run ID, number of cells and latest step from the robot database without modifying it.

    Raises sqlite3.OperationalError if the database is locked against reading.
    """
    try:
        with contextlib.closing(sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=1)) as conn:
            cursor = conn.cursor()
            run_id = cursor.execute("SELECT `value` FROM Settings_Table WHERE `key` = 'Base Sample ID'").fetchone()
            (cells,) = cursor.execute("SELECT COUNT(*) FROM Cell_Assembly_Table WHERE `Cell Number` > 0").fetchone()
            active, finished = cursor.execute(
                "SELECT COUNT(*), COALESCE(SUM(`Last Completed Step` >= ?), 0) FROM Cell_Assembly_Table "
                "WHERE `Cell Number` > 0 AND `Error Code` = 0",
                (FINAL_STEP,),
            ).fetchone()
            latest = cursor.execute(
                "SELECT `Cell Number`, `Step Number` FROM Timestamp_Table ORDER BY `Timestamp` DESC LIMIT 1",
            ).fetchone()
    except sqlite3.Error as e:
        if is_lock_error(e):
            raise
        return RunSummary(error=f"Database not available: {e}")

    summary = RunSummary(run_id=str(run_id[0]) if run_id else "-", cells=cells)
    if active and finished == active:
        summary.current_step = "Run complete"
    elif latest is not None:
        cell, step = latest
        step_info = config.STEP_DEFINITION.get(int(step), {}) if step is not None else {}
        summary.current_step = f"Cell {cell}, current step: {step_info.get('Description', 'Unknown step')}"
    return summary


class SummaryPoller:
    """Reads the run summary and checks for a stuck database lock in a background thread.

    locked becomes True once the write lock has been held by someone else for lock_warning_s.
    """

    def __init__(self, db_path: Path, interval_s: float = 2.0, lock_warning_s: float = 4.0) -> None:
        """Set the database, how often to read it and how long a lock lasts before warning."""
        self.db_path = db_path
        self.interval_s = interval_s
        self.lock_warning_s = lock_warning_s
        self.summary = RunSummary(current_step="Reading database...")
        self.locked = False
        self._locked_since: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="summary-poller", daemon=True)

    def poll(self) -> None:
        """Check the lock and read the summary, keeping the previous summary while reads are blocked."""
        if database_write_locked(self.db_path):
            now = time.monotonic()
            self._locked_since = self._locked_since if self._locked_since is not None else now
            locked = now - self._locked_since >= self.lock_warning_s
        else:
            self._locked_since = None
            locked = False
        if locked and not self.locked:
            logger.warning("Database is locked, the robot cannot write to it")
        elif self.locked and not locked:
            logger.info("Database is no longer locked")
        self.locked = locked
        try:
            self.summary = read_run_summary(self.db_path)
        except sqlite3.Error:
            logger.debug("Database locked against reading, keeping last summary")

    def start(self) -> None:
        """Start polling."""
        self._thread.start()

    def stop(self) -> None:
        """Stop polling."""
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll()
            self._stop.wait(self.interval_s)


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to fit inside width x height, keeping the aspect ratio."""
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


class CameraTile:
    """Live preview and status of one camera."""

    def __init__(self, parent: tk.Misc, station: "Station") -> None:
        """Build the tile widgets."""
        self.station = station
        self.frame = ttk.LabelFrame(parent, text=f"{station.name.capitalize()} camera", padding=4)
        self.image_area = ttk.Frame(self.frame)
        self.image_area.pack(fill="both", expand=True)
        self.image_area.pack_propagate(False)  # noqa: FBT003  Image size follows the tile, not the other way round
        self.image_label = ttk.Label(self.image_area, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self._photo: ImageTk.PhotoImage | None = None
        self._shown: tuple[int, int, int] | None = None
        self._no_signal = False

    def refresh(self) -> None:
        """Show the latest preview if there is a new frame or the tile was resized."""
        camera = self.station.camera
        if not camera.connected:
            if not self._no_signal:
                self._no_signal = True
                self._shown = None
                self._photo = None
                self.image_label.configure(image="", text="No signal", foreground="grey")
            return
        self._no_signal = False
        preview = camera.preview()
        width, height = self.image_area.winfo_width(), self.image_area.winfo_height()
        if preview is None or width < 10 or height < 10 or self._shown == (preview.frame_id, width, height):
            return
        self._shown = (preview.frame_id, width, height)
        image = fit_image(preview.image, width, height)
        if self.station.target is not None:  # Drawn after resizing so the lines stay sharp
            image = self.station.target.overlay(image, preview.scale * image.shape[1] / preview.image.shape[1])
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.image_label.configure(image=self._photo, text="")


class CameraViewer:
    """Main window with the run summary, a tile per camera and recent activity."""

    def __init__(self, stations: dict[str, "Station"], columns: int = 2, refresh_ms: int = 50) -> None:
        """Build the window, call run() to show it."""
        self.refresh_ms = refresh_ms
        self.root = tk.Tk()
        self.root.title("Aurora robot cameras")
        self.root.geometry("1280x900")
        self.root.protocol("WM_DELETE_WINDOW", self.confirm_quit)
        self.root.bind("<Escape>", lambda _e: self.confirm_quit())
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(10, 8))
        header.grid(row=0, column=0, sticky="ew")
        self.lock_banner = tk.Label(
            header,
            text="DATABASE LOCKED! The robot cannot write to it. Save or discard changes in any open SQLite viewer.",
            background="#c62828",
            foreground="white",
            font=("Segoe UI", 14, "bold"),
            pady=6,
        )
        self.run_label = ttk.Label(header, font=("Segoe UI", 11))
        self.run_label.pack(anchor="w")
        self.step_label = ttk.Label(header, font=("Segoe UI", 14, "bold"))
        self.step_label.pack(anchor="w")

        grid = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        grid.grid(row=1, column=0, sticky="nsew")
        self.tiles = [CameraTile(grid, station) for station in stations.values()]
        for i, tile in enumerate(self.tiles):
            tile.frame.grid(row=i // columns, column=i % columns, sticky="nsew", padx=4, pady=4)

        n = len(self.tiles)
        activity = ttk.LabelFrame(grid, text="Activity", padding=4)
        activity.grid(row=n // columns, column=n % columns, columnspan=columns - n % columns, sticky="nsew", padx=4)
        self.activity = tk.Text(activity, height=8, wrap="none", state="disabled", font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(activity, command=self.activity.yview)
        self.activity.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.activity.pack(fill="both", expand=True)
        for column in range(columns):
            grid.columnconfigure(column, weight=1, uniform="tile")
        for row in range(n // columns + 1):
            grid.rowconfigure(row, weight=1, uniform="tile")

        self.poller = SummaryPoller(config.DATABASE_FILEPATH)
        self._log_queue: queue.Queue[logging.LogRecord] = queue.Queue()
        self._log_handler = QueueHandler(self._log_queue)
        self._log_handler.setLevel(logging.INFO)

    def confirm_quit(self) -> None:
        """Ask before closing, the robot cannot take photos while the daemon is stopped."""
        if messagebox.askokcancel(
            "Stop cameras?",
            "The robot cannot take photos while the cameras are stopped.",
            parent=self.root,
        ):
            self.root.quit()

    def refresh(self) -> None:
        """Update the summary, camera tiles and activity log."""
        banner_shown = bool(self.lock_banner.winfo_manager())
        if self.poller.locked and not banner_shown:
            self.lock_banner.pack(before=self.run_label, fill="x", pady=(0, 6))
        elif not self.poller.locked and banner_shown:
            self.lock_banner.pack_forget()
        summary = self.poller.summary
        if summary.error:
            self.run_label.configure(text=summary.error, foreground="red")
        else:
            self.run_label.configure(text=f"Run: {summary.run_id}    Cells: {summary.cells}", foreground="")
        self.step_label.configure(text=summary.current_step)
        for tile in self.tiles:
            tile.refresh()
        self._drain_activity()

    def _drain_activity(self) -> None:
        lines = []
        with contextlib.suppress(queue.Empty):
            while True:
                record = self._log_queue.get_nowait()
                time_str = logging.Formatter().formatTime(record, "%H:%M:%S")
                lines.append(f"{time_str}  {record.getMessage()}\n")
        if not lines:
            return
        self.activity.configure(state="normal")
        self.activity.insert("end", "".join(lines))
        excess = int(self.activity.index("end-1c").split(".")[0]) - MAX_ACTIVITY_LINES
        if excess > 0:
            self.activity.delete("1.0", f"{excess + 1}.0")
        self.activity.see("end")
        self.activity.configure(state="disabled")

    def _tick(self) -> None:
        self.refresh()
        self.root.after(self.refresh_ms, self._tick)

    def run(self) -> None:
        """Show the window until it is closed."""
        activity_logger = logging.getLogger(ACTIVITY_LOGGER)
        previous_level = activity_logger.level
        if activity_logger.getEffectiveLevel() > logging.INFO:
            activity_logger.setLevel(logging.INFO)
        activity_logger.addHandler(self._log_handler)
        previous_sigint = signal.signal(signal.SIGINT, lambda *_: self.root.quit())
        self.poller.start()
        try:
            self._tick()
            self.root.mainloop()
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
            self.poller.stop()
            activity_logger.removeHandler(self._log_handler)
            activity_logger.setLevel(previous_level)
            with contextlib.suppress(tk.TclError):
                self.root.destroy()
