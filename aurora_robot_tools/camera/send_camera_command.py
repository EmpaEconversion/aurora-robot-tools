"""Send commands to the camera daemon."""

import socket
from time import sleep
from typing import Literal

from aurora_robot_tools.config import CAMERA_PORT


def send_command(command: Literal["capturebottom", "capturetop", "capturebottomqr"]) -> None:
    """Trigger camera to record snapshot."""
    PORT = CAMERA_PORT
    # To allow for the camera to adjust exposure
    if command == "capturetop":
        sleep(5)
    if command in ["capturebottom", "capturebottomqr"]:
        sleep(0.5)

    # Connect to camera daemon and send command
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.connect(("127.0.0.1", PORT))
        except ConnectionRefusedError:
            print("Camera daemon not running - start with 'aurora-rt startcam.")
        client.sendall(command.encode())
        # Wait for a response
        response = client.recv(1024).decode()
        print(f"Response from camera daemon: {response}")
        sleep(0.5)  # Prevent robot moving too quickly after taking a photo
