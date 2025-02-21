"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Assigns cell numbers to presses.

Data is read from the Cell_Assembly_Table and Press_Table tables in the chemspeedDB database. The
script identifies the cells and presses that are available for assembly, and assigns the cells to
presses by updating both tables. The AutoSuite software will then use this information to assemble
the appropriate cells.

Usage:
    The script is called from assign_cells_to_press.exe by the AutoSuite software.
    It can also be called from the command line.

    There are two optional parameters that can set with the command line call.

    - `link_rack_pos_to_press` (bool, default False):
        If set, press 1 will only accept cells from rack positions 1, 7, 13, 19, 25, 31. Press 2
        only accepts cells from rack positions 2, 8, 14, 20, 26, 32, and so on.

    - `limit_electrolytes_per_batch` (int, default 0):
        0 - No limit on the number of different electrolytes in a batch of up to 6 cells.
        n (integer) - Limit the number of different electrolytes in a batch of up to 6 cells to n.
            This is useful if the electrolyte is volatile, since the cleaning step between each 
            electrolyte switch is time-consuming.

    e.g. `py assign_cells_to_press.py 1 2`
    This will ensure that rack positions and press positions are linked (rack 1 only goes to press 
    1, rack 2 to press 2, etc.) and limit the number of different electrolytes in each batch to 2.
"""

import sqlite3
import sys
from tkinter import Tk, messagebox

import numpy as np
import pandas as pd

DATABASE_FILEPATH = "C:\\Modules\\Database\\chemspeedDB.db"

RETURN_STEP = 140  # Step number for returned cell in robot recipe

link_rack_pos_to_press = bool(sys.argv[1]) if len(sys.argv) >= 2 else True
limit_electrolytes_per_batch = int(sys.argv[2]) if len(sys.argv) >= 3 else 0

PRESS_TO_RACK = {
    1 : 1,
    2 : 4,
    3 : 2,
    4 : 5,
    5 : 3,
    6 : 6,
}

with sqlite3.connect(DATABASE_FILEPATH) as conn:
    # Read the table Cell_Assembly_Table and Press_Table from the database
    df = pd.read_sql("SELECT * FROM Cell_Assembly_Table", conn)
    df_press = pd.read_sql("SELECT * FROM Press_Table", conn)

    # Check where the cell number loaded is 0 and where the error code is 0 for the presses
    working_press_numbers = np.where(df_press["Error Code"] == 0)[0]+1

    # Find rack positions with cells that are assigned for assembly (Cell Number > 0), have not
    # finished assembly, with no error code, and find their cell numbers and electrolyte positions
    available_rack_pos = np.where(
        (df["Cell Number"]>0) &
        (df["Last Completed Step"]<RETURN_STEP) &
        (df["Error Code"]==0) &
        (df["Current Press Number"]==0),
        )[0]+1
    available_cell_numbers = df.loc[available_rack_pos-1, "Cell Number"].to_numpy().astype(int)
    available_electrolytes = df.loc[available_rack_pos-1, "Electrolyte Position"].to_numpy().astype(int)

    if link_rack_pos_to_press:
        print("Using link_rack_pos_to_press")
    if limit_electrolytes_per_batch:
        print(f"Limiting electrolytes to {limit_electrolytes_per_batch} per batch")

    electrolytes_used = []
    presses_with_errors = df_press.loc[df_press["Error Code"]!=0, "Press Number"].to_numpy()
    presses_already_loaded = df.loc[df["Current Press Number"]>0, "Current Press Number"].to_numpy()
    cells_already_loaded = df.loc[df["Current Press Number"]>0, "Cell Number"].to_numpy()
    rack_already_loaded = df.loc[df["Current Press Number"]>0, "Rack Position"].to_numpy()
    presses_to_load = []
    cells_to_load = []
    rack_to_load = []

    # Loop through presses, check conditions then assign the first available cell to the press
    for press in range(1,7):
        availability_mask = np.ones(len(available_rack_pos), dtype=bool)

        # If no more cells available, stop
        if available_cell_numbers.size == 0:
            print("No more cells available")
            break

        # If using link_rack_pos_to_press and press has an error code,
        # add an error code to all rack positions linked to that press
        if (press in presses_with_errors) and link_rack_pos_to_press:
            error_mask = (available_rack_pos-1)%6+1 == PRESS_TO_RACK[press]
            if available_cell_numbers[error_mask].size>0:
                print(f"Press {press} has an error, "
                        f"giving error code to cells with rack position {available_cell_numbers[error_mask]}")
                df.loc[available_rack_pos[error_mask]-1, "Error Code"] = 301
            else:
                print(f"Press {press} has an error")
            continue

        # If press already has a cell loaded
        if press in presses_already_loaded:
            idxs = df.loc[df["Current Press Number"] == press].index
            error_msg = (f'Press {press} has a cell already loaded.\n'
                      'Check "Current Press Number" column in cell_assembly_table in the database.')
            if len(idxs) != 1:
                raise ValueError(error_msg)
            # If there is no error, add the electrolyte to the list of used electrolytes
            if df["Error Code"].loc[idxs[0]] == 0:
                electrolyte = df["Electrolyte Position"].loc[idxs[0]]
                electrolytes_used.append(electrolyte)
            continue

        # If using link_rack_pos_to_press, only consider cells in the correct rack position
        if link_rack_pos_to_press:
            availability_mask = (available_rack_pos-1)%6+1 == PRESS_TO_RACK[press]

        # Only allow limit_electrolytes_per_batch different electrolytes to be loaded at once (if > 0)
        if limit_electrolytes_per_batch and len(set(electrolytes_used)) >= limit_electrolytes_per_batch:
            availability_mask &= [electrolyte in electrolytes_used for electrolyte in available_electrolytes]

        # Assign the first available cell to the press
        final_available_cell_numbers = available_cell_numbers[availability_mask]
        if final_available_cell_numbers.size > 0:
            loaded_cell=final_available_cell_numbers[0]
            cells_to_load.append(loaded_cell)
            presses_to_load.append(press)
            rack_to_load.append(available_rack_pos[availability_mask][0])
            if limit_electrolytes_per_batch:
                electrolytes_used.append(loaded_cell)
            df_press.loc[press-1, "Current Cell Number Loaded"] = loaded_cell
            df.loc[df["Cell Number"]==loaded_cell, "Current Press Number"] = press

            # Remove the loaded cell from the available cells
            removed_idx = np.where(available_cell_numbers==loaded_cell)[0][0]
            available_cell_numbers = np.delete(available_cell_numbers, removed_idx)
            available_rack_pos = np.delete(available_rack_pos, removed_idx)
            available_electrolytes = np.delete(available_electrolytes, removed_idx)
        else:
            print(f"Press {press} has no available cells to load")
            continue

    # If there are cells already loaded into presses and new cells that can be loaded
    # ask the user if they want to start assembling new cells
    if len(presses_already_loaded) > 0 and len(cells_to_load) > 0:
        root = Tk()
        root.withdraw()
        load_new_cells = messagebox.askyesno(
            title="Cells already loaded",
            message=
            "Some cells are already loaded into presses:\n\nPress | Rack | Cell\n"
            + "".join([f"{p:<10} {r:<9} {c:<9}\n" for p, r, c in
                       zip(presses_already_loaded,rack_already_loaded,cells_already_loaded)]) +
            "\nDo you also want to load new cells?\n\nPress | Rack | Cell\n"
            + "".join([f"{p:<10} {r:<9} {c:<9}\n" for p, r, c in
                       zip(presses_to_load, rack_to_load, cells_to_load)]),
        )
    else:
        load_new_cells=True

    # Write the updated tables back to the database
    if load_new_cells and len(cells_to_load) > 0:
        print("Loading:\n"+"Press | Rack | Cell\n"+
              "".join([f"{p:<7} {r:<6} {c:<6}\n" for p, r, c in zip(presses_to_load, rack_to_load, cells_to_load)]))
        df_press.to_sql("Press_Table", conn, index=False, if_exists="replace")
        df.to_sql("Cell_Assembly_Table", conn, index=False, if_exists="replace")
        print("Successfully updated the database")
    elif len(cells_to_load) == 0:
        print("No cells available to load")
    else:
        print("Not loading new cells - finishing current assembly first")
