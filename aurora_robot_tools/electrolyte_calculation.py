"""Copyright © 2025, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Determine the electrolyte mixing steps required.

The script reads the electrolytes required and their volumes from the Cell_Assembly_Table in the
chemspeedDB database. It then reads the mixing ratios from the Electrolyte_Table and calculates all
the of the mixing steps (such as move 100 uL from vial 1 to vial 5, etc.) required to prepare all
electrolytes for the cells.

Usage:
    The script is called from electrolyte_calculation.exe by the AutoSuite software.
    It can also be called from the command line.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aurora_robot_tools.config import DATABASE_FILEPATH


def read_db(db_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Cell_Assembly_Table and Electrolyte_Table from the database."""
    with sqlite3.connect(db_path) as conn:
        # Read the tables from the database
        df = pd.read_sql("SELECT * FROM Cell_Assembly_Table", conn)
        df_electrolyte = pd.read_sql("SELECT * FROM Electrolyte_Table", conn)
    return df, df_electrolyte


def get_mix_fractions(df_electrolyte: pd.DataFrame) -> np.ndarray:
    """Get a square matrix of the mixture fractions."""
    # Initialise square matrix
    n = df_electrolyte["Electrolyte Position"].max()
    mix_fractions = np.zeros((n, n))
    # Fill from electrolyte table
    for i in range(n):
        mix_fractions[:, i] = df_electrolyte[f"Mix {i + 1}"]
    # Make sure no nans and rows are normalised
    mix_fractions = np.nan_to_num(mix_fractions)
    for i in range(n):
        if mix_fractions[i].sum() != 0:
            mix_fractions[i] = mix_fractions[i] / mix_fractions[i].sum()  # normalise the row
    return mix_fractions


def get_volumnes(
    df: pd.DataFrame,
    mix_fractions: np.ndarray,
    safety_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the volumes of electrolyte required.

    Cumulative volumes account for the electrolyte being used up in the mixing steps.
    """
    n = len(mix_fractions)
    volumes = np.zeros(n)
    for i in range(n):
        mask = (df["Electrolyte Position"] == i + 1) & (df["Cell Number"] > 0) & (df["Error Code"] == 0)
        volumes[i] = df.loc[mask, "Electrolyte Amount (uL)"].sum() * safety_factor
    cumulative_volumes = volumes
    remaining_volumes = volumes
    for _ in range(5):
        remaining_volumes = np.matmul(remaining_volumes, mix_fractions)
        cumulative_volumes = cumulative_volumes + remaining_volumes
    return volumes, cumulative_volumes


def make_mixing_steps(mixing_matrix: np.ndarray) -> pd.DataFrame:
    """Create dataframe containing list of mixing steps.

    The steps are ordered such that each vial must have completed all of its steps before it is used
    as a source for another vial.
    """
    source_positions = []
    target_positions = []
    volumes_to_mix = []
    for j in range(len(mixing_matrix)):
        for i in range(len(mixing_matrix)):
            if mixing_matrix[i, j] > 0:
                source_positions.append(j + 1)
                target_positions.append(i + 1)
                volumes_to_mix.append(mixing_matrix[i, j])

    # Write the mixing steps to a dataframe
    return pd.DataFrame(
        columns=["Source Position", "Target Position", "Volume (uL)"],
        data=zip(source_positions, target_positions, volumes_to_mix, strict=True),
    )


def write_db(db_path: Path, df_electrolyte: pd.DataFrame, df_mixing_table: pd.DataFrame) -> None:
    """Write the electrolyte and mixing table back to the database."""
    with sqlite3.connect(db_path) as conn:
        df_electrolyte.to_sql("Electrolyte_Table", conn, index=False, if_exists="replace")
        df_mixing_table.to_sql(
            "Mixing_Table",
            conn,
            index=False,
            if_exists="replace",
            dtype={
                "Target Position": "INTEGER",
                "Source Position": "INTEGER",
                "Volume (uL)": "REAL",
            },
        )


def main(safety_factor: float = 1.1) -> None:
    """Determine the electrolyte mixing steps."""
    print(f"Multiplying all electrolyte volumes by {safety_factor}.")

    df, df_electrolyte = read_db(DATABASE_FILEPATH)

    mix_fractions = get_mix_fractions(df_electrolyte)

    # Calculate the volumes of electrolyte required
    volumes, cumulative_volumes = get_volumnes(df, mix_fractions, safety_factor)

    # Add these to the electrolyte table
    df_electrolyte["Volume Required (uL)"] = volumes
    df_electrolyte["Cumulative Volume Required (uL)"] = cumulative_volumes

    # Create a matrix with the volumes required for each source and target vial
    mixing_matrix = mix_fractions * volumes[:, np.newaxis]

    # Create the list of mixing steps
    df_mixing_table = make_mixing_steps(mixing_matrix)

    # Write the electrolyte and mixing table back to the database
    write_db(DATABASE_FILEPATH, df_electrolyte, df_mixing_table)

    print("Successfully calculated the electrolyte mixing steps, wrote to Mixing_Table in database.")


if __name__ == "__main__":
    safety_factor = float(sys.argv[1]) if len(sys.argv) >= 2 else 1.1
    main(safety_factor)
