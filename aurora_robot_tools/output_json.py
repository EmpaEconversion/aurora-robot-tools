"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Convert the finished database to a JSON file to go to aurora_cycler_manager.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

import pandas as pd
import pytz

from aurora_robot_tools import config

PRESS_STEP = next(k for k, v in config.STEP_DEFINITION.items() if v["Step"] == "Press")


def read_db(db_path: Path, press_step: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Read completed cells, timestamps, and run_id from robot database."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            f"SELECT * FROM Cell_Assembly_Table WHERE `Last Completed Step` >= {press_step} AND `Error Code` = 0",
            conn,
        )
        df_timestamp = pd.read_sql(
            "SELECT * FROM Timestamp_Table WHERE `Complete` = 1",
            conn,
        )
        cursor = conn.cursor()
        cursor.execute("SELECT `value` FROM Settings_Table WHERE `key` = 'Base Sample ID'")
        run_id = cursor.fetchone()[0]
    df["Run ID"] = run_id
    return df, df_timestamp, run_id


def user_output_filepath(default_folder: Path, run_id: str) -> Path:
    """Ask user where to save the JSON file."""
    # Open file dialog to set the output file path
    Tk().withdraw()  # to hide the main window
    output_filepath = Path(
        filedialog.asksaveasfilename(
            title="Export chemspeed.db to .json",
            filetypes=[("json files", "*.json")],
            initialdir=default_folder,
            initialfile=f"{run_id}.json",
        ),
    )
    # check if it is a valid excel file
    if not output_filepath.name:
        output_filepath = Path(config.OUTPUT_DIR) / f"{run_id}.json"
    if output_filepath.suffix != ".json":
        output_filepath = output_filepath.with_suffix(".json")
    return output_filepath


def generate_assembly_history(timestamps: pd.Series) -> list:
    """Take a row of timestamps, turn into a list of dicts describing assembly history."""
    history = []
    timestamp_dict = timestamps.to_dict()
    for i in config.STEP_DEFINITION:
        # check if key exists
        step: dict[str, str | int] = {}
        ts = timestamp_dict.get(i)
        if ts and isinstance(ts, str):
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S %z")
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
                except ValueError:
                    dt = datetime.strptime(ts, "%d.%m.%Y %H:%M")  # noqa: DTZ007
            dt = pytz.timezone(config.TIME_ZONE).localize(dt)
            step["Step"] = config.STEP_DEFINITION[i]["Step"]
            step["Description"] = config.STEP_DEFINITION[i]["Description"]
            step["Timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S %z")
            step["uts"] = int(dt.timestamp())
            history.append(step)
    return history


def generate_all_assembly_history(df: pd.DataFrame, df_timestamp: pd.DataFrame) -> pd.DataFrame:
    """Generate assembly history for all cells using the timestamp table."""
    # Drop nans, cast to int, sort by timestamp, drop duplicates
    df_timestamp = df_timestamp.dropna()
    df_timestamp["Step Number"] = df_timestamp["Step Number"].astype(int)
    df_timestamp["Cell Number"] = df_timestamp["Cell Number"].astype(int)
    df_timestamp = df_timestamp.sort_values("Timestamp", ascending=False)
    df_timestamp = df_timestamp.drop_duplicates(["Cell Number", "Step Number"])
    # Pivot the table so that each step number is a column
    df_timestamp = df_timestamp.pivot_table(
        index="Cell Number",
        columns="Step Number",
        values="Timestamp",
        aggfunc="last",
    )
    # Create a Assembly history column for each cell number
    df_timestamp["Assembly History"] = df_timestamp.apply(generate_assembly_history, axis=1)
    # Merge assembly history into the cell assembly table on cell number
    return df.merge(df_timestamp["Assembly History"], on="Cell Number")


def main() -> None:
    """Export sample details from robot database to a JSON file."""
    # Read db
    df, df_timestamp, run_id = read_db(config.DATABASE_FILEPATH, PRESS_STEP)

    # Ask user for output file path
    output_filepath = user_output_filepath(config.OUTPUT_DIR, run_id)

    # If df is empty (no finished cells), exit
    if df.empty:
        print("No finished cells found in database. No output file created.")
        sys.exit()

    # Remove certain columns, these are either unnecessary or will be recalculated
    columns_to_drop = [
        "Last Completed Step",
        "Current Press Number",
        "Error Code",
        "Anode Balancing Capacity (mAh)",
        "Cathode Balancing Capacity (mAh)",
        "N:P ratio overlap factor",
        "N:P Ratio",
        "N:P Ratio Maximum",
        "N:P Ratio Minimum",
        "N:P Ratio Target",
    ]
    df = df.drop(columns=columns_to_drop)

    # Generate the assembly history list[dict] for all cells
    df = generate_all_assembly_history(df, df_timestamp)

    # Output the file
    df.to_json(output_filepath, orient="records", indent=4)


if __name__ == "__main__":
    main()
