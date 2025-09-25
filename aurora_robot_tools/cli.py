"""Command line interface for robot tools."""

import logging
from typing import Annotated

from typer import Argument, Option, Typer

v_option = Option(0, "--verbose", "-v", count=True, help="Increase verbosity")
q_option = Option(0, "--quiet", "-q", count=True, help="Decrease verbosity")

app = Typer(
    add_completion=False,
    pretty_exceptions_enable=False,
)


def get_log_level(verbosity: int, quietness: int) -> int:
    """Get the logging level based on verbosity and quietness."""
    level = logging.WARNING  # Default level
    level -= verbosity * 10
    level += quietness * 10
    level = max(level, logging.DEBUG)
    level = min(level, logging.CRITICAL)
    return level


@app.command()
def import_excel() -> None:
    """Import excel file and load into robot database."""
    from aurora_robot_tools.import_excel import main as import_excel_main

    import_excel_main()


@app.command()
def electrolyte(safety_factor: float = Argument(1.1)) -> None:
    """Determine electrolyte mixing steps."""
    from aurora_robot_tools.electrolyte_calculation import main as electrolyte_main

    electrolyte_main(safety_factor)


@app.command()
def backup() -> None:
    """Backup the robot database."""
    from aurora_robot_tools.backup_database import main as backup_main

    backup_main()


@app.command()
def balance(mode: int = Argument(6)) -> None:
    """Perform electrode balancing."""
    from aurora_robot_tools.capacity_balance import main as balance_main

    balance_main(mode)


@app.command()
def assign(link: bool = Argument(True), elyte_limit: int = Argument(0)) -> None:  # noqa: FBT003
    """Assign cells to presses."""
    from aurora_robot_tools.assign_cells_to_press import main as assign_main

    assign_main(link, elyte_limit)


@app.command()
def startcam(verbosity: int = v_option, quietness: int = q_option) -> None:
    """Start the camera daemon."""
    from aurora_robot_tools.camera.camera_daemon import logger as startcam_logger
    from aurora_robot_tools.camera.camera_daemon import main as startcam_main

    log_level = get_log_level(verbosity, quietness)
    startcam_logger.setLevel(log_level)
    startcam_logger.addHandler(logging.StreamHandler())
    startcam_main()


@app.command()
def top_photo() -> None:
    """Save a photo of the pressing tools."""
    from aurora_robot_tools.camera.send_camera_command import send_command

    send_command("capturetop")


@app.command()
def bottom_photo() -> None:
    """Save a photo from the bottom-up alignment camera."""
    from aurora_robot_tools.camera.send_camera_command import send_command

    send_command("capturebottom")


@app.command()
def bottom_photo_qr() -> None:
    """Save a photo from bottom-up camera, detect QR code, write to database."""
    from aurora_robot_tools.camera.send_camera_command import send_command

    send_command("capturebottomqr")


@app.command()
def output() -> None:
    """Output the robot database to a JSON file."""
    from aurora_robot_tools.output_json import main as output_main

    output_main()


@app.command()
def led(setting: str) -> None:
    """Set the LED ring light color."""
    from aurora_robot_tools.camera.ringlight import set_light

    set_light(setting)


@app.command()
def find_circles(
    folder: str = Argument(None),
) -> None:
    """Find circles in images."""
    from pathlib import Path

    from aurora_robot_tools.camera.alignment import process_folder

    # If no folder path is provided, use the current working directory
    folder_path = Path.cwd() if folder is None else Path(folder)
    process_folder(folder_path)


@app.command()
def recalibrate(
    app_path: str,
    calibration_path: Annotated[list[str] | None, Argument()] = None,
    fit_to_grid: bool = True,
) -> None:
    """Recalibrate the APP file."""
    from aurora_robot_tools.chemapp_edit import realign_app

    realign_app(app_path, calibration_path, fit_to_grid)


@app.command()
def app_to_xml(filepath: str) -> None:
    """Convert Chemspeed APP to XML."""
    from aurora_robot_tools.chemapp_edit import app_to_xml

    app_to_xml(filepath)


@app.command()
def xml_to_app(filepath: str) -> None:
    """Convert Chemspeed APP to XML."""
    from aurora_robot_tools.chemapp_edit import xml_to_app

    xml_to_app(filepath)


if __name__ == "__main__":
    app()
