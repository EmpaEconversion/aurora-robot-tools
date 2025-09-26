"""Open chemspeed.app files and edit stuff hehehehe."""

import gzip
import sqlite3
from pathlib import Path

import lxml.etree as et
import numpy as np
import pandas as pd
import xmltodict
from scipy.optimize import least_squares


def app_to_xml(filepath: Path | str) -> Path:
    """Gzip open the .app, save as .xml."""
    filepath = Path(filepath)
    with gzip.open(filepath, "rb") as f1, filepath.with_suffix(".xml").open("wb") as f2:
        f2.write(f1.read())
    return filepath.with_suffix(".xml")


def xml_to_app(filepath: Path | str) -> Path:
    """Gzip the .xml file, save as .app."""
    filepath = Path(filepath)
    with filepath.open("rb") as f1, gzip.open(filepath.with_suffix(".app"), "wb") as f2:
        f2.write(f1.read())
    return filepath.with_suffix(".app")


class ChemspeedApp:
    """Class to open and edit chemspeed.app files."""

    def __init__(self, filepath: Path | str) -> None:
        """Open a chemspeed.app file and store as element tree."""
        self.filepath = Path(filepath)
        self.tree = self.openfile(Path(filepath))
        self.racks = self.get_all_racks()

    def openfile(self, filepath: Path) -> et.ElementTree:
        """Open a chemspeed.app file and return the root element."""
        with gzip.open(filepath, "rb") as f:
            return et.parse(f)

    def get_all_racks(self) -> dict[str, et.Element]:
        """Get the rack elements from the xml tree."""
        elements = self.tree.findall(".//*[@typeid='Chemspeed.SAModuleConfigurableRack.1']")
        racks = {}
        for element in elements:
            name = element.find("name").text
            racks[name] = element
        return racks

    def get_rack(self, rack_name: str) -> dict[str, str | dict]:
        """Get dictionary of a rack."""
        rack = self.racks.get(rack_name)
        if rack is None:
            msg = f"Rack {rack_name} not found."
            raise ValueError(msg)
        wells = rack.find(".//wellparameterss")
        return xmltodict.parse(et.tostring(wells))["wellparameterss"]

    def get_wells(self, rack: str | dict) -> np.ndarray:
        """Get the x and y values from a rack dict."""
        coords = []
        if isinstance(rack, str):
            rack = self.get_rack(rack)
        assert isinstance(rack, dict)  # noqa: S101
        for i in range(int(rack["count"])):
            x = float(rack[f"wellparameters{i}"]["xvalue"])
            y = float(rack[f"wellparameters{i}"]["yvalue"])
            coords.append([x, y])
        return np.array(coords)

    def write_rack_wells(self, rack_name: str, coords: np.ndarray) -> None:
        """Write new coordinates back to the xml."""
        rack = self.racks.get(rack_name)
        if rack is None:
            msg = f"Rack {rack_name} not found."
            raise ValueError(msg)
        count = int(rack.find("wellparameterss/count").text)
        if coords.shape[0] != count:
            msg = f"Number of coordinates {coords.shape[0]} does not match rack count {count}."
            raise ValueError(msg)
        for i in range(count):
            x = str(coords[i][0])
            y = str(coords[i][1])
            rack.find(f"wellparameterss/wellparameters{i}/xvalue").text = x
            rack.find(f"wellparameterss/wellparameters{i}/yvalue").text = y

    def save_as(self, filepath: Path | str) -> None:
        """Write the xml tree to a file."""
        with gzip.open(filepath, "wb") as f:
            self.tree.write(f, encoding="utf-8", xml_declaration=True)


def get_bottom_rack_idx(rack_pos: int) -> int:
    """Get the index for the bottom rack from a rack position."""
    if rack_pos < 1 or rack_pos > 18:
        msg = "Rack position must be between 1 and 18 for bottom half."
        raise ValueError(msg)
    return (rack_pos - 1) % 2 * 9 + (rack_pos - 1) // 2


def get_top_rack_idx(rack_pos: int) -> int:
    """Get the index for the top rack from a rack position."""
    if rack_pos < 19 or rack_pos > 36:
        msg = "Rack position must be between 19 and 36 for top half."
        raise ValueError(msg)
    return (rack_pos - 1) % 2 * 9 + (rack_pos - 19) // 2


def get_full_rack_idx(rack_pos: int) -> int:
    """Get the index for a full rack from a rack position."""
    if rack_pos < 1 or rack_pos > 36:
        msg = "Rack position must be between 1 and 36."
        raise ValueError(msg)
    return (rack_pos - 1) % 2 * 18 + (rack_pos - 1) // 2


def rectangular_grid(x0: float, dx: float, y0: float, dy: float, theta: float, nx: int = 2, ny: int = 9) -> np.ndarray:
    """Create a rectangular grid of points."""
    # Get square grid
    i = np.arange(nx).reshape(nx, 1)
    j = np.arange(ny).reshape(1, ny)
    x = x0 + i * dx  # shape (nx, 1)
    y = y0 + j * dy  # shape (1, ny)
    # Apply rotation
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    grid = np.stack((x_rot, y_rot), axis=2)
    # reshape to (nx*ny, 2)
    return grid.reshape(nx * ny, 2)


def fit_coords_to_grid(coords: np.ndarray, nx: int = 2, ny: int = 9) -> tuple:
    """Fit measured coordinates to a grid."""

    def residuals(params: tuple) -> np.ndarray:
        """Residuals are euclidean distance between measured and grid coordinates."""
        x0, dx, y0, dy, theta = params
        grid = rectangular_grid(x0, dx, y0, dy, theta, nx, ny)
        diff = coords - grid
        residuals = np.linalg.norm(diff, axis=1)
        residuals[np.isnan(coords).any(axis=1)] = 0
        return residuals

    # initial guess
    init_params = np.array([0.016, 0.023, 0.016, 0.023, 0])

    result = least_squares(residuals, init_params)

    new_coords = rectangular_grid(
        result.x[0],
        result.x[1],
        result.x[2],
        result.x[3],
        result.x[4],
        nx=nx,
        ny=ny,
    )

    return new_coords, result


def get_alignment_from_db(filepath: Path | str) -> pd.DataFrame:
    """Read the calibration database."""
    filepath = Path(filepath)
    with sqlite3.connect(filepath) as conn:
        return pd.read_sql(
            "SELECT * FROM Calibration_Table",
            conn,
        )


def get_alignment_from_json(filepath: Path | str) -> pd.DataFrame:
    """Get the alignment from a json file."""
    return pd.read_json(filepath)


step_dict = {
    "Bottom": {
        "Step": 10,
        "Full rack": "Bottom casing (36 well)",
        "Gripper type": "1NH",
    },
    "Anode": {
        "Step": 30,
        "Bottom rack": "Anode Bottom (18 well)",
        "Top rack": "Anode Top (18 well)",
        "Gripper type": "1NH",
    },
    "Cathode": {
        "Step": 40,
        "Bottom rack": "Cathode Bottom (18 well)",
        "Top rack": "Cathode Top (18 well)",
        "Gripper type": "1NH",
    },
    "Separator": {
        "Step": 60,
        "Bottom rack": "Separator Bottom (18 well)",
        "Top rack": "Separator Top (18 well)",
        "Gripper type": "1NH",
    },
    "Spacer": {
        "Step": 100,
        "Bottom rack": "Spacer Bottom (18 well)",
        "Top rack": "Spacer Top (18 well)",
        "Gripper type": "1NH",
    },
    "Top": {
        "Step": 120,
        "Full rack": "Top casing (36 well)",
        "Gripper type": "1NH",
    },
}


def realign_app(
    app_path: Path | str,
    calibration_path: Path | str | list[str] | list[Path] | None = None,
    fit_to_grid: bool = True,
) -> None:
    """Recalibrate the APP file."""
    app_path = Path(app_path)
    if calibration_path is None:
        calibration_path = []
    if not isinstance(calibration_path, list):
        calibration_path = [calibration_path]  # type: ignore  # noqa: PGH003
    assert isinstance(calibration_path, list)  # noqa: S101
    calibration_path = [Path(p) for p in calibration_path]
    if len(calibration_path) == 0:
        # look for alignment json in current working directory
        calibration_path = list(Path.cwd().glob("*alignment*.json"))
        if len(calibration_path) == 0:
            msg = "No calibration files found. Please provide a path to a calibration file."
            raise ValueError(msg)

    # build dataframe from all calibration files
    dfs = []
    for cpath in calibration_path:
        if cpath.suffix == ".db":
            dfs.append(get_alignment_from_db(cpath))
        elif cpath.suffix == ".json":
            dfs.append(get_alignment_from_json(cpath))
        else:
            print(f"Unknown file type {cpath.suffix}. Skipping.")
    df = pd.concat(dfs, ignore_index=True)

    if "Rack Position" not in df.columns:
        df["Rack Position"] = df["Cell Number"]

    reference = df[(df["Step Number"] == 0) & (df["Rack Position"] == 0)]
    if reference.empty:
        print("WARNING: No reference point found. Assuming camera is perfectly aligned.")
    else:
        # Make everything relative to the reference point
        print(
            "Found a reference with coordinates "
            f"{reference['dx_mm'].to_numpy()[0]:.4f}, {reference['dy_mm'].to_numpy()[0]:.4f} mm"
        )
        df["dx_mm"] -= reference["dx_mm"].to_numpy()[0]
        df["dy_mm"] -= reference["dy_mm"].to_numpy()[0]

    # open the app file
    myapp = ChemspeedApp(app_path)

    wells_before = []
    wells_after = []
    wells_after_fit = []
    rack_names = []
    for component_name, component_dict in step_dict.items():
        # Filter the dataframe for the component
        step = component_dict["Step"]
        fdf = df[df["Step Number"] == step]
        print(f"Found {len(fdf)} points for {component_name} with step {step}.")
        if fdf.empty:
            print(f"No alignment found for {component_name}.")
            continue
        if len(fdf) < 4:
            print(f"Not enough points for {component_name}. Need at least 4 to align.")
            continue

        # Get the gripper type for the component
        gripper_type = component_dict.get("Gripper type")
        if gripper_type not in ["1NH", "4NH"]:
            msg = f"Unknown gripper type {gripper_type} for {component_name}."
            raise ValueError(msg)

        # Align all racks asscociated with the component
        for rack_type in ["Bottom rack", "Top rack", "Full rack"]:
            # Get the rack name and find it in the app
            rack_name = component_dict.get(rack_type)
            if rack_name is not None:
                assert isinstance(rack_name, str)  # noqa: S101
                wells = myapp.get_wells(myapp.get_rack(rack_name))
                wells_orig = wells.copy()
                wells_edited = np.empty_like(wells)
                wells_edited[:, :] = np.nan

                # Filter the dataframe for the rack type
                if rack_type == "Bottom rack":
                    ffdf = fdf[fdf["Rack Position"] < 19]
                elif rack_type == "Top rack":
                    ffdf = fdf[fdf["Rack Position"] > 18]
                elif rack_type == "Full rack":
                    ffdf = fdf
                else:
                    msg = f"Unknown rack type {rack_type} for {component_name}."
                    raise ValueError(msg)

                # Reject if not enough points to calibrate
                if len(ffdf) < 4:
                    print(f"Not enough points for {component_name}. Need at least 4 to align.")
                    continue

                # Adjust the coordinates based on the alignment
                # In photo:
                # +dx means the PIECE is too far in +x, so the PIECE needs to move -x
                # +dy means the PIECE is too far in +y, so the PIECE needs to move -y

                # Bottom rack 4NH:
                # to move PIECE +x move the 4NH +y
                # to move PIECE +y move the 4NH -x

                # Top rack 4NH:
                # to move PIECE +x move the 4NH -y
                # to move PIECE +y move the 4NH +x

                # Any rack 1NH:
                # to move PIECE +x move the 1NH -x
                # to move PIECE +y move the 1NH -y

                for _i, row in ffdf.iterrows():
                    rack_pos = int(row["Rack Position"])
                    dx_mm = row["dx_mm"]
                    dy_mm = row["dy_mm"]

                    # Get the index
                    get_index = {
                        "Bottom rack": get_bottom_rack_idx,
                        "Top rack": get_top_rack_idx,
                        "Full rack": get_full_rack_idx,
                    }
                    idx = get_index[rack_type](rack_pos)

                    if gripper_type == "1NH":
                        # Doesn't matter if the rack is top, bottom, or full

                        # the piece is +dx_mm too far in x
                        # to correct we move the piece -dx_mm in x
                        # so move the 1NH +dx_mm in x
                        wells[idx][0] += dx_mm / 1000  # 1NH pickup x
                        wells_edited[idx][0] = wells[idx][0].copy()
                        # the piece is +dy_mm too far in y
                        # to correct we move the piece -dy_mm in y
                        # we need to move the 1NH +dy_mm in x
                        wells[idx][1] += dy_mm / 1000  # 1NH pickup y
                        wells_edited[idx][1] = wells[idx][1].copy()

                    elif gripper_type == "4NH":
                        # 4NH faces in different directions for top and bottom
                        # so needs different corrections

                        if rack_type == "Full rack":
                            msg = f"Rack '{rack_name}': 4NH cannot be used with full rack"
                            raise ValueError(msg)

                        if rack_type == "Bottom rack":
                            # the piece is +dx_mm too far in x
                            # to correct we move the piece -dx_mm in x
                            # so move the 4NH -dx_mm in y
                            wells[idx][1] -= dx_mm / 1000  # 4NH pickup y
                            wells_edited[idx][1] = wells[idx][1].copy()
                            # the piece is +dy_mm too far in y
                            # to correct we move the piece -dy_mm in y
                            # we need to move the 4NH +dy_mm in x
                            wells[idx][0] += dy_mm / 1000  # 4NH pickup x
                            wells_edited[idx][0] = wells[idx][0].copy()

                        elif rack_type == "Top rack":
                            # the piece is +dx_mm too far in x
                            # to correct we move the piece -dx_mm in x
                            # we need to move the 4NH +dx_mm in y
                            wells[idx][1] += dx_mm / 1000  # 4NH pickup y
                            wells_edited[idx][1] = wells[idx][1].copy()
                            # the piece is +dy_mm too far in y
                            # to correct we move the piece -dy_mm in y
                            # we need to move the 4NH -dy_mm in x
                            wells[idx][0] -= dy_mm / 1000  # 4NH pickup x
                            wells_edited[idx][0] = wells[idx][0].copy()

                # Fit to a rectangular grid if needed
                if fit_to_grid:
                    print(f"Fitting {rack_name} to grid based on {len(ffdf)} points.")
                    if rack_type == "Full rack":
                        wells, _res = fit_coords_to_grid(wells_edited, 2, 18)
                    else:
                        wells, _res = fit_coords_to_grid(wells_edited, 2, 9)
                wells_before.append(wells_orig)
                wells_after.append(wells_edited)
                wells_after_fit.append(wells)
                rack_names.append(rack_name)

                # Write back to the app xml
                myapp.write_rack_wells(rack_name, wells)
                # size//2 because there are 2 coordinates for each well
                print(f"Updated {wells.size // 2} wells in {rack_name}.")

    # Save the new app file
    new_filename = app_path.with_name(app_path.stem + "_calibrated.app")
    myapp.save_as(new_filename)
    print(f"Saved calibrated app file to {new_filename}")
