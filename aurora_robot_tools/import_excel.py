"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Read the user input Excel file and write the data to a SQL database.

Users should use the prepared Excel template to input all of the parameters of the cells to be
assembled. The script reads the file, manipulates, and writes to the Chemspeed database which is
used by the AutoSuite software to assemble the cells.

Usage:
    Run file directly, use the CLI, or call from Autosuite software.
"""

import sqlite3
import warnings
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd

from aurora_robot_tools.config import DATABASE_FILEPATH, INPUT_DIR

# Ignore the pandas data validation warning
warnings.filterwarnings("ignore", ".*extension is not supported and will be removed.*")


def get_input(default: str | Path) -> Path:
    """Open a dialog to select the input file."""
    Tk().withdraw()  # to hide the main window
    file_path = Path(
        filedialog.askopenfilename(
            initialdir=default,
            title="Select the input Excel file",
            filetypes=[("Excel files", "*.xlsx")],
        ),
    )
    # check if it is a valid excel file
    if not file_path.exists():
        msg = "No file selected."
        raise ValueError(msg)
    if file_path.suffix != ".xlsx":
        msg = "Selected file is not a .xlsx file."
        raise ValueError(msg)
    return file_path


def read_excel(input_filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read excel file, return as main, component and electrolyte dataframes."""
    try:
        df = pd.read_excel(
            input_filepath,
            sheet_name="Input Table",
            dtype={"Bottom Spacer Type": str, "Top Spacer Type": str},
        )
        df_components = pd.read_excel(
            input_filepath,
            sheet_name="Component Properties",
        )
        df_electrolyte = pd.read_excel(
            input_filepath,
            sheet_name="Electrolyte Properties",
            skiprows=1,
        )
    except ValueError:
        print("CRITICAL: Excel file format not correct. Check your input file and try again.")
        raise
    return df, df_components, df_electrolyte


def create_aux_tables(input_filepath: Path) -> pd.DataFrame:
    """Create the press, settings and timestamp tables."""
    df_press = pd.DataFrame()
    df_press["Press Number"] = [1, 2, 3, 4, 5, 6]
    df_press["Current Cell Number Loaded"] = 0
    df_press["Error Code"] = 0
    df_press["Last Completed Step"] = 0

    df_settings = pd.DataFrame()
    df_settings["key"] = ["Input Filepath", "Base Sample ID"]
    df_settings["value"] = [str(input_filepath), str(input_filepath.stem)]

    df_timestamp = pd.DataFrame(columns=["Cell Number", "Step Number", "Timestamp", "Complete"])

    return df_press, df_settings, df_timestamp


def merge_electrolyte(df: pd.DataFrame, df_electrolyte: pd.DataFrame) -> pd.DataFrame:
    """Merge electrolyte details into the main dataframe based on electrolyte position."""
    df["Electrolyte Name"] = df["Electrolyte Position"].map(
        df_electrolyte.set_index("Electrolyte Position")["Name"],
    )
    df["Electrolyte Description"] = df["Electrolyte Position"].map(
        df_electrolyte.set_index("Electrolyte Position")["Description"],
    )
    df["Electrolyte Amount (uL)"] = (
        df["Electrolyte Amount Before Separator (uL)"] + df["Electrolyte Amount After Separator (uL)"]
    )
    return df


def merge_electrodes(df: pd.DataFrame, df_components: pd.DataFrame) -> pd.DataFrame:
    """Merge electrode details into the main dataframe based on electrode type."""
    # df_anode is df_electrodes where 'anode' is in the column name
    df_anode = df_components[[col for col in df_components.columns if "Anode" in col]]
    df_anode = df_anode.dropna(subset=["Anode Type"])
    # if diameter is missing or 0, set to 15 mm
    df_anode["Anode Diameter (mm)"] = df_anode["Anode Diameter (mm)"].fillna(15).replace(0, 15)

    # df_cathode is df_electrodes where 'cathode' is in the column name
    df_cathode = df_components[[col for col in df_components.columns if "Cathode" in col]]
    df_cathode = df_cathode.dropna(subset=["Cathode Type"])
    # if diameter is missing or 0, set to 14 mm
    df_cathode["Cathode Diameter (mm)"] = df_cathode["Cathode Diameter (mm)"].fillna(14).replace(0, 14)

    # If Anode Type or Cathode Type contains duplicates, raise an error
    if df_anode["Anode Type"].duplicated().any() or df_cathode["Cathode Type"].duplicated().any():
        msg = (
            "CRITICAL: Anode Type or Cathode Type in electrode properties table contains "
            "duplicates. Check the input file.",
        )
        raise ValueError(msg)

    # Merge Anode and Cathode into table
    df = df.merge(df_anode, on="Anode Type", how="left")
    df = df.merge(df_cathode, on="Cathode Type", how="left")
    return df


def merge_other_components(df: pd.DataFrame, df_components: pd.DataFrame) -> pd.DataFrame:
    """Merge in details of separator, casing, and spacer."""
    # Merge separator into table
    df_separator = df_components[[col for col in df_components.columns if "Separator" in col]].dropna()
    df = df.merge(df_separator, on="Separator Type", how="left")

    # Merge casing into table
    df_casing = df_components[[col for col in df_components.columns if "Casing" in col]].dropna()
    df = df.merge(df_casing, on="Casing Type", how="left")

    # Merge spacer into table
    df_spacer = df_components[[col for col in df_components.columns if "Spacer" in col]]
    for spacer_pos in ["Top", "Bottom"]:
        df_spacer_specific = df_spacer.rename(
            columns={col: f"{spacer_pos} {col}" for col in df_spacer.columns},
        ).dropna()
        df = df.merge(df_spacer_specific, on=f"{spacer_pos} Spacer Type", how="left")
        df[f"{spacer_pos} Spacer Thickness (mm)"] = df[f"{spacer_pos} Spacer Thickness (mm)"].fillna(0)
    return df


def add_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns which will be filled in by the robot later."""
    df["Anode Mass (mg)"] = 0
    df["Anode Active Material Mass (mg)"] = 0
    df["Anode Balancing Capacity (mAh)"] = 0
    df["Anode Rack Position"] = 0
    df["Cathode Mass (mg)"] = 0
    df["Cathode Active Material Mass (mg)"] = 0
    df["Cathode Balancing Capacity (mAh)"] = 0
    df["Cathode Rack Position"] = 0
    df["N:P ratio overlap factor"] = 0
    df["N:P Ratio"] = 0
    df["Cell Number"] = 0
    df["Last Completed Step"] = 0
    df["Current Press Number"] = 0
    df["Error Code"] = 0
    df["Barcode"] = ""
    df["Sample ID"] = ""

    # First filling of anode and cathode positions
    df.loc[df["Anode Type"].notna(), "Anode Rack Position"] = df["Rack Position"]
    df.loc[df["Cathode Type"].notna(), "Cathode Rack Position"] = df["Rack Position"]
    df.loc[df["Anode Type"].notna(), "Anode Mass (mg)"] = 0
    df.loc[df["Cathode Type"].notna(), "Cathode Mass (mg)"] = 0

    return df


def reorder_df(df: pd.DataFrame) -> pd.DataFrame:
    """Re-order columns in database, put frequently used at start, otherwise alphabetical."""
    columns = df.columns.tolist()
    first_cols = [
        "Rack Position",
        "Cell Number",
        "Current Press Number",
        "Last Completed Step",
        "Error Code",
        "Comments",
    ]
    for f in first_cols:
        columns.remove(f)
    columns = first_cols + sorted(columns)
    return df[columns]


def sanity_check(df: pd.DataFrame) -> None:
    """Check columns are present and have sensible values."""
    # Warnings to the user
    columns_to_check = [
        "Anode Type",
        "Anode Balancing Specific Capacity (mAh/g)",
        "Anode C-rate Definition Specific Capacity (mAh/g)",
        "Anode C-rate Definition Areal Capacity (mAh/cm2)",
        "Cathode Type",
        "Cathode Balancing Specific Capacity (mAh/g)",
        "Cathode C-rate Definition Specific Capacity (mAh/g)",
        "Cathode C-rate Definition Areal Capacity (mAh/cm2)",
        "N:P Ratio Target",
        "N:P Ratio Minimum",
        "N:P Ratio Maximum",
        "Separator Type",
        "Electrolyte Position",
        "Electrolyte Amount (uL)",
        "Electrolyte Amount Before Separator (uL)",
        "Electrolyte Amount After Separator (uL)",
        "Batch Number",
    ]
    missing_columns = set(columns_to_check) - set(df.columns)
    if missing_columns:
        msg = f"CRITICAL: these columns are missing from the input: {', '.join(missing_columns)}"
        raise ValueError(msg)

    if (df["Electrolyte Amount (uL)"] > 500).any():
        msg = (
            f"CRITICAL: Your input has electrolyte volumes that are too large: {max(df['Electrolyte Amount (uL)'])} uL."
        )
        raise ValueError(msg)

    if (df["Electrolyte Amount (uL)"] > 150).any():
        print(
            f"WARNING: your input has large electrolyte volumes up to {max(df['Electrolyte Amount (uL)'])} uL.",
        )

    if any(df["Rack Position"].to_numpy() != np.arange(1, 37)):
        msg = "CRITICAL: Rack positions must be sequential 1-36. Check the input file."
        raise ValueError(msg)

    if any((df["Top Spacer Thickness (mm)"] + df["Bottom Spacer Thickness (mm)"]) > 2.0):
        msg = "CRITICAL: Too much spacer! For safety reasons you can only have <= 2.0 mm total."
        raise ValueError(msg)

    if any(df["Top Spacer Thickness (mm)"] < 0) or any(df["Bottom Spacer Thickness (mm)"] < 0):
        msg = "CRITICAL: Negative valued spacer thickness."
        raise ValueError(msg)

    if any(df["Separator Thickness (mm)"] > 1.0):
        msg = "CRITICAL: You have separators thicker than 1 mm, this is not currently allowed."
        raise ValueError(msg)


def write_to_sql(
    db_path: Path,
    df: pd.DataFrame,
    df_press: pd.DataFrame,
    df_electrolyte: pd.DataFrame,
    df_settings: pd.DataFrame,
    df_timestamp: pd.DataFrame,
) -> None:
    """Write the dataframes to an SQLite3 database to be used by the robot."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql(
            "Cell_Assembly_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype={
                "Anode Rack Position": "INTEGER",
                "Cathode Rack Position": "INTEGER",
                "Cell Number": "INTEGER",
                "Last Completed Step": "INTEGER",
                "Current Press Number": "INTEGER",
                "Error Code": "INTEGER",
                "Casing Type": "TEXT",
                "Barcode": "TEXT",
                "Batch Number": "INTEGER",
            },
        )
        df_press.to_sql(
            "Press_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype=dict.fromkeys(df_press.columns, "INTEGER"),
        )
        electrolyte_dtype = dict.fromkeys(df_electrolyte.columns, "REAL")
        electrolyte_dtype["Electrolyte Position"] = "INTEGER"
        electrolyte_dtype["Name"] = "TEXT"
        electrolyte_dtype["Description"] = "TEXT"
        df_electrolyte.to_sql(
            "Electrolyte_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype=electrolyte_dtype,
        )
        df_settings.to_sql(
            "Settings_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype={"key": "TEXT", "value": "TEXT"},
        )
        df_timestamp.to_sql(
            "Timestamp_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype={
                "Cell Number": "INTEGER",
                "Step Number": "INTEGER",
                "Timestamp": "VARCHAR(255)",
                "Complete": "BOOLEAN",
            },
        )
        df_calibration = pd.DataFrame(
            columns=["Cell Number", "Step Number", "dx_mm", "dy_mm"],
        )
        df_calibration.to_sql(
            "Calibration_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype={
                "Cell Number": "INTEGER",
                "Step Number": "INTEGER",
                "Rack Position": "INTEGER",
                "dx_mm": "REAL",
                "dy_mm": "REAL",
            },
        )


def main() -> None:
    """Read in excel input, manipulate, and write to sql database."""
    input_filepath = get_input(INPUT_DIR)
    df, df_components, df_electrolyte = read_excel(input_filepath)
    df_press, df_settings, df_timestamp = create_aux_tables(input_filepath)
    df = merge_electrolyte(df, df_electrolyte)
    df = merge_electrodes(df, df_components)
    df = merge_other_components(df, df_components)
    df = add_extra_columns(df)
    df = reorder_df(df)
    print("Successfully read and manipulated the Excel file.")
    sanity_check(df)
    write_to_sql(Path(DATABASE_FILEPATH), df, df_press, df_electrolyte, df_settings, df_timestamp)
    print("Successfully updated the database.")


if __name__ == "__main__":
    main()
