"""Common configuration settings for the Aurora robot tools."""

from pathlib import Path

DATABASE_FILEPATH = Path("C:/Modules/Database/chemspeedDB.db")
DATABASE_BACKUP_DIR = Path("C:/Modules/Database/Backup/")
TIME_ZONE = "Europe/Zurich"
INPUT_DIR = Path("%userprofile%/Desktop/Inputs/")
OUTPUT_DIR = Path("%userprofile%/Desktop/Outputs/")
IMAGE_DIR = Path("C:/Aurora_images/")

CAMERA_PORT = 13865

MM_TO_PX = 1500 / 20

# Current step definitions
STEP_DEFINITION: dict[int, dict[str, str | float]] = {
    0: {
        "Step": "Unknown",
        "Description": "Step performed without step number",
        "Radius": 0.5,
    },
    1: {
        "Step": "4NH reference",
        "Description": "Calibration of 4NH tool",
        "Radius": 0.5,
    },
    2: {
        "Step": "1NH reference",
        "Description": "Calibration of 1NH tool",
        "Radius": 0.5,
    },
    3: {
        "Step": "Press 4NH reference",
        "Description": "Calibration of pressing position with 4NH",
        "Radius": 10.0,
    },
    4: {
        "Step": "Press gripper reference",
        "Description": "Calibration of pressing position with gripper tool",
        "Radius": 10.0,
    },
    5: {
        "Step": "Press gripper QR reference",
        "Description": "Calibration of pressing position with gripper tool at QR zone",
        "Radius": 10.0,
    },
    6: {
        "Step": "Press 1NH reference",
        "Description": "Calibration of pressing position with 1NH",
        "Radius": 10.0,
    },
    10: {
        "Step": "Bottom",
        "Description": "Place bottom casing",
        "Radius": 10.0,
    },
    20: {
        "Step": "Spacer",
        "Description": "Place bottom spacer",
        "Radius": 8.0,
    },
    30: {
        "Step": "Anode",
        "Description": "Place anode face up",
        "Radius": 7.5,
    },
    40: {
        "Step": "Cathode",
        "Description": "Place cathode face up",
        "Radius": 7.0,
    },
    50: {
        "Step": "Electrolyte",
        "Description": "Add electrolyte before separator",
    },
    60: {
        "Step": "Separator",
        "Description": "Place separator",
        "Radius": 8.0,
    },
    70: {
        "Step": "Electrolyte",
        "Description": "Add electrolyte after separator",
    },
    80: {
        "Step": "Anode",
        "Description": "Place anode face down",
        "Radius": 7.5,
    },
    90: {
        "Step": "Cathode",
        "Description": "Place cathode face down",
        "Radius": 7.0,
    },
    100: {
        "Step": "Spacer",
        "Description": "Place top spacer",
        "Radius": 8.0,
    },
    110: {
        "Step": "Spring",
        "Description": "Place spring",
    },
    120: {
        "Step": "Top",
        "Description": "Place top casing",
        "Radius": 9.0,
    },
    130: {
        "Step": "Press",
        "Description": "Press cell using 7.8 kN hydraulic press",
    },
    140: {
        "Step": "Return",
        "Description": "Return completed cell to rack",
    },
}

# Step definitions from robot tools 0.1.x
STEP_DEFINITION_0_1 = {
    1: {
        "Step": "Bottom",
        "Description": "Place bottom casing",
    },
    2: {
        "Step": "Anode",
        "Description": "Place anode face up",
    },
    3: {
        "Step": "Electrolyte",
        "Description": "Add electrolyte before separator",
    },
    4: {
        "Step": "Separator",
        "Description": "Place separator",
    },
    5: {
        "Step": "Electrolyte",
        "Description": "Add electrolyte after separator",
    },
    6: {
        "Step": "Cathode",
        "Description": "Place cathode face down",
    },
    7: {
        "Step": "Spacer",
        "Description": "Place top spacer",
    },
    8: {
        "Step": "Spring",
        "Description": "Place spring",
    },
    9: {
        "Step": "Top",
        "Description": "Place top casing",
    },
    10: {
        "Step": "Press",
        "Description": "Press cell using 7.8 kN hydraulic press",
    },
    11: {
        "Step": "Return",
        "Description": "Return completed cell to rack",
    },
}

# Define zone name : [bottom_rack_name, top_rack_name], or [full_rack_name]
ZONE_ORDERING: dict[str, list[str]] = {
    "Anode Zone": ["Anode Bottom (18 well)", "Anode Top (18 well)"],
    "Cathode Zone": ["Cathode Bottom (18 well)", "Cathode Top (18 well)"],
    "Separator Zone": ["Separator Bottom (18 well)", "Separator Top (18 well)"],
    "Spacer Zone": ["Spacer Bottom (18 well)", "Spacer Top (18 well)"],
    "Bottom Zone": ["Bottom casing (36 well)"],
    "Top Zone": ["Top casing (36 well)"],
    "Spring Zone": ["Spring (36 well)"],
    "Gasket Zone": ["Top casing (36 well)"],
}
