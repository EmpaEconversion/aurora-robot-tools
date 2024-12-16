"""Copyright © 2024, Empa, Graham Kimbell, Enea Svaluto-Ferro, Ruben Kuhnel, Corsin Battaglia.

Match cathodes with anodes to achieve the desired N:P ratio.

The script reads the weights measured by the cell assembly robot, along with other input parameters,
from the Cell_Assembly_Table in the chemspeedDB database. The preferred method calculates every
possible N:P ratio from all anodes and cathode combinations, then uses the linear sum assignment
algorithm to find the optimal matching of anodes and cathodes. The script then writes the updated
table back to the database, which is used by the AutoSuite software to assemble the cells.

The electrode matching is done in batches (defined in the input excel table), so only electrodes
within the same batch are switched around. This is useful if there are different cell chemistries
within one run of the robot.

Note: currently the script only moves the cathodes and not the anode positions. This means that each
anode is tied to its target N:P ratio, so the sorting is not optimal if the user requires different
N:P ratios within one batch of cells.

Usage:
    The script is called from capacity_balance.exe, which is called from the AutoSuite software.
    It can also be called from the command line.

    There is one additional parameter that can be set:

    - `sorting_method`:
        0 - Do not sort, do not check N:P ratio
                Use if you want to keep the anodes and cathodes in the same order as they are
                e.g. when restarting the program part-way through a run
        1 - Do not sort the anodes and cathodes
                Not recommended
        2 - Sort the anodes and cathodes by capacity
                Suboptimal, not recommended
        3 - Use the 2D cost matrix method
                Optimal if N:P ratios are the same within batches
        4 - Use greedy 3D matching
                Suboptimal, only use if N:P ratios differ and exact 3D is too slow
        5 - Use exact 3D matching
                Optimal if N:P ratios differ within batches, but can be slow
        6 - Choose automatically (default)
                If N:P ratios do not change, use 2D matching (method 3), otherwise try exact 3D
                (method 5), if too slow use greedy 3D (method 4)

Todo:
    - Make rejection_cost_factor an argument when AutoSuite supports it.
    - [Long term] Pre-calculate the possible matchings using different rejection_cost_factors and
      allow the user to choose the best one.

"""
import itertools
import sqlite3
import sys

import numpy as np
import pandas as pd
import pulp
from scipy.optimize import linear_sum_assignment

DATABASE_FILEPATH = "C:\\Modules\\Database\\chemspeedDB.db"

TIMEOUT_SECONDS = 30


def calculate_capacity(df: pd.DataFrame) -> None:
    """Calculate the capacity of the anodes and cathodes in-place in the main dataframe, df.

    Args:
        df (pandas.DataFrame): The dataframe containing the cell assembly data.

    """
    for xode in ["Anode", "Cathode"]:
        df[f"{xode} Active Material Weight (mg)"] = (
            (df[f"{xode} Weight (mg)"] - df[f"{xode} Current Collector Weight (mg)"])
            * df[f"{xode} Active Material Weight Fraction"]
        )
        df[f"{xode} Balancing Capacity (mAh)"] = (
            1e-3 * df[f"{xode} Active Material Weight (mg)"] * df[f"{xode} Balancing Specific Capacity (mAh/g)"]
        )
        if (df[f"{xode} Balancing Capacity (mAh)"] < 0).any():
            print(f"WARNING: {xode} capacities below 0, setting to NaN")
            df.loc[df[f"{xode} Balancing Capacity (mAh)"] < 0, f"{xode} Balancing Capacity (mAh)"] = np.nan


def cost_matrix_assign(df: pd.DataFrame, rejection_cost_factor: float = 2) -> tuple[list[int], list[int]]:
    """Calculate the cost matrix and find the optimal matching of anodes and cathodes.

    Args:
        df (pandas.DataFrame): The dataframe containing the cell assembly data.
        rejection_cost_factor (float, optional): cost of rejected cells. Defaults to 2.
            1 = no extra cost for rejecting, more rejected cells, better N:P ratio of accepted cells
            10 = high cost to reject cells, fewer rejected cells, worse N:P ratio of accepted cells
            2 = compromise

    Returns:
        tuple: The indices of the optimal matching of anodes and cathodes.

    """
    # Calculate all possible N:P ratios
    actual_ratio = np.outer(
        df["Anode Balancing Capacity (mAh)"] / df["Anode Diameter (mm)"]**2,
        1 / (df["Cathode Balancing Capacity (mAh)"] / df["Cathode Diameter (mm)"]**2),
    )
    n = actual_ratio.shape[0]

    # Cells outside N:P ratio limits are rejected, given the same cost scaled by rejection_cost_factor
    for i in range(n):
        actual_ratio[i, actual_ratio[i] > df["Maximum N:P Ratio"].iloc[i]] = (
            df["Maximum N:P Ratio"].iloc[i] * rejection_cost_factor
        )
        actual_ratio[i, actual_ratio[i] < df["Minimum N:P Ratio"].iloc[i]] = (
            df["Minimum N:P Ratio"].iloc[i] / rejection_cost_factor
        )

    # Calculate the cost matrix
    cost_matrix = np.abs(actual_ratio - np.outer(df["Target N:P Ratio"], np.ones(n)))

    # Prefer unassigned cathodes to not swap with each other
    # by making nans on the diagonal cost very slightly less
    for i in range(n):
        if np.isnan(cost_matrix[i, i]):
            cost_matrix[i, i] = 999.99999999
    # otherwise unassigned cells have the same cost
    cost_matrix = np.nan_to_num(cost_matrix, nan=1000)

    # Find the optimal matching of anodes and cathodes using linear sum assignment
    anode_ind, cathode_ind = linear_sum_assignment(cost_matrix, maximize=False)

    return anode_ind, cathode_ind


def exact_npartite_matching(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the optimal matching of anodes and cathodes using an exact 3D matching algorithm.

    This algorithm his NP-hard and can take a very long time for n>10
    """
    # Get the size of the cost matrix
    n = cost_matrix.shape[0]

    # Create a list of all possible assignments (each assignment is a tuple of 3 indices)
    assignments = list(itertools.product(range(n), repeat=3))

    # Create a binary variable for each assignment
    x = pulp.LpVariable.dicts("x", assignments, cat=pulp.LpBinary)

    # Create the problem
    problem = pulp.LpProblem("3D_assignment", pulp.LpMinimize)

    # The objective is to minimize the total cost of the chosen assignments
    problem += pulp.lpSum(cost_matrix[a]*x[a] for a in assignments)

    # Add constraints ensuring each x, y, and z is used exactly once
    for i in range(n):
        problem += pulp.lpSum(x[a] for a in assignments if a[0] == i) == 1
        problem += pulp.lpSum(x[a] for a in assignments if a[1] == i) == 1
        problem += pulp.lpSum(x[a] for a in assignments if a[2] == i) == 1

    # Solve the problem
    print(f"Attempting exact matching, will give up if a solution not found in {TIMEOUT_SECONDS} seconds...")
    problem.solve(pulp.PULP_CBC_CMD(options=[f"sec={TIMEOUT_SECONDS}"],msg=False))
    if pulp.LpStatus[problem.status] != "Optimal":
        msg = f"Optimal solution not found. Status: {pulp.LpStatus[problem.status]}"
        raise ValueError(msg)
    print("Optimal solution found")
    # Get the optimal assignments
    optimal_assignments = np.array([a for a in assignments if pulp.value(x[a]) == 1])
    i_idx, j_idx, k_idx = optimal_assignments[:,0], optimal_assignments[:,1], optimal_assignments[:,2]
    return i_idx, j_idx, k_idx


def greedy_npartite_matching(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the optimal matching of anodes, cathodes, and target ratios using a greedy algorithm.

    This will find a suboptimal solution, but does not suffer from combinatoral explosion like the
    exact method.
    """
    # Get the shape of the cost matrix
    n = cost_matrix.shape[0]

    # Create a list of all possible assignments (each assignment is a tuple of 3 indices)
    assignments = [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]

    # Sort the assignments by cost
    assignments.sort(key=lambda a: cost_matrix[a])

    # Initialize the set of used indices for each dimension
    used_indices = [set() for _ in range(3)]

    # Initialize the list of chosen assignments
    chosen_assignments = []

    # Iterate over the sorted assignments
    for a in assignments:
        # If the indices of this assignment have not been used yet, choose this assignment
        if all(a[i] not in used_indices[i] for i in range(3)):
            chosen_assignments.append(a)
            for i in range(3):
                used_indices[i].add(a[i])
    chosen_assignments = np.array(chosen_assignments)
    i_idx, j_idx, k_idx = chosen_assignments[:,0], chosen_assignments[:,1], chosen_assignments[:,2]
    return i_idx, j_idx, k_idx


def cost_matrix_assign_3d(
        df: pd.DataFrame,
        rejection_cost_factor: float = 2,
        exact: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the cost matrix and find optimal matching with 3D algorithm.

    Args:
        df (pandas.DataFrame): The dataframe containing the cell assembly data.
        rejection_cost_factor (float, optional): cost of rejected cells. Defaults to 2.
            1 = no extra cost for rejecting, more rejected cells, better N:P ratio of accepted cells
            10 = high cost to reject cells, fewer rejected cells, worse N:P ratio of accepted cells
            2 = compromise
        exact (bool, optional): Use exact matching. Defaults to False.

    Returns:
        tuple: The indices of the optimal matching of anodes and cathodes.

    """
    n = len(df)

    # Convert all 1D arrays to 3D n x n x n arrays
    anode_capacity = np.array(df["Anode Balancing Capacity (mAh)"]/df["Anode Diameter (mm)"]**2)
    anode_capacity = np.tile(anode_capacity[:, np.newaxis, np.newaxis], (1, n, n))
    cathode_capacity = np.array(df["Cathode Balancing Capacity (mAh)"]/df["Cathode Diameter (mm)"]**2)
    cathode_capacity = np.tile(cathode_capacity[np.newaxis, :, np.newaxis], (n, 1, n))
    target_ratio = np.array(df["Target N:P Ratio"])
    target_ratio = np.tile(target_ratio[np.newaxis, np.newaxis, :], (n, n, 1))
    min_ratio = np.array(df["Minimum N:P Ratio"])
    min_ratio = np.tile(min_ratio[np.newaxis, np.newaxis, :], (n, n, 1))
    max_ratio = np.array(df["Maximum N:P Ratio"])
    max_ratio = np.tile(max_ratio[np.newaxis, np.newaxis, :], (n, n, 1))

    # Calculate the 3D cost matrix
    cost_matrix = anode_capacity/cathode_capacity - target_ratio

    # If the cost diff is negative, divide by (min_ratio - target_ratio)
    neg_mask = cost_matrix < 0
    cost_matrix[neg_mask] = cost_matrix[neg_mask]/(min_ratio[neg_mask] - target_ratio[neg_mask])

    # If the cost diff is positive, divide by (max_ratio - target_ratio)
    pos_mask = cost_matrix > 0
    cost_matrix[pos_mask] = cost_matrix[pos_mask]/(max_ratio[pos_mask] - target_ratio[pos_mask])

    # If the normalised cost is over 1 the cell is rejected, so set the cost to the rejection_cost_factor
    cost_matrix[cost_matrix > 1] = rejection_cost_factor

    # Set NaNs to a very large number, diagonal elements slightly less so unassigned electrodes are not moved
    for i in range(n):
        if np.isnan(cost_matrix[i, i, i]):
            cost_matrix[i, i, i] = 999.999
    cost_matrix = np.nan_to_num(cost_matrix, nan=1000)

    # Find the optimal matching of anodes and cathodes using greedy algorithm
    if exact:
        anode_ind, cathode_ind, ratio_ind = exact_npartite_matching(cost_matrix)
    else:
        anode_ind, cathode_ind, ratio_ind = greedy_npartite_matching(cost_matrix)

    # Sort such that the anode doesn't change order
    ind_sort=np.argsort(anode_ind)
    return anode_ind[ind_sort], cathode_ind[ind_sort], ratio_ind[ind_sort]


def rearrange_electrode_columns(
        df: pd.DataFrame,
        row_indices: np.ndarray,
        anode_ind: np.ndarray,
        cathode_ind: np.ndarray,
        ratio_ind: np.ndarray,
    ) -> None:
    """Rearrange eletrode columns in-place in the main dataframe.

    Args:
        df (pandas.DataFrame): The dataframe containing the cell assembly data.
        row_indices (numpy.ndarray): The indices for the rows in df being rearranged.
        anode_ind (numpy.ndarray): Anode indices for optimal matching.
        cathode_ind (numpy.ndarray): Cathode indices for optimal matching.
        ratio_ind (numpy.ndarray): Ratio indices for optimal matching.

    """
    anode_columns = [col for col in df.columns if "Anode" in col]
    cathode_columns = [col for col in df.columns if "Cathode" in col]
    ratio_columns = ["Target N:P Ratio", "Minimum N:P Ratio", "Maximum N:P Ratio"]
    df_immutable = df.copy()
    for column in anode_columns:
        df.loc[row_indices, column] = df_immutable.loc[row_indices[anode_ind], column].to_numpy()
    for column in cathode_columns:
        df.loc[row_indices, column] = df_immutable.loc[row_indices[cathode_ind], column].to_numpy()
    for column in ratio_columns:
        df.loc[row_indices, column] = df_immutable.loc[row_indices[ratio_ind], column].to_numpy()
    # Recalculate N:P ratio overlap factor
    df["N:P ratio overlap factor"] = (df["Cathode Diameter (mm)"]**2 / df["Anode Diameter (mm)"]**2).fillna(0)


def update_cell_numbers(df: pd.DataFrame, base_sample_id: str, check_NP_ratio: bool = True) -> None:
    """Update the cell numbers in the main dataframe, df, based on the accepted cells.

    Args:
        df (pandas.DataFrame): The dataframe containing the cell assembly data.
        base_sample_id (str): The run ID for the cells.
        check_NP_ratio (bool, optional): Check the N:P ratio. Defaults to True.
    """
    if check_NP_ratio:
        df["Actual N:P Ratio"] = (
            (df["Anode Balancing Capacity (mAh)"] / df["Anode Diameter (mm)"]**2) /
            (df["Cathode Balancing Capacity (mAh)"] / df["Cathode Diameter (mm)"]**2)
        )
        cell_meets_criteria = ((df["Actual N:P Ratio"] >= df["Minimum N:P Ratio"])
                                & (df["Actual N:P Ratio"] <= df["Maximum N:P Ratio"]))
        accepted_cell_indices = np.where(cell_meets_criteria)[0]
        rejected_cell_indices = np.where(~cell_meets_criteria & ~df["Actual N:P Ratio"].isna())[0]
        average_deviation = np.mean(np.abs(df["Actual N:P Ratio"][accepted_cell_indices]
                                        - df["Target N:P Ratio"][accepted_cell_indices]))
        print(f"Accepted {len(accepted_cell_indices)} cells "
            f"with average N:P deviation from target: {average_deviation:.4f}\n"
            f"Rejected {len(rejected_cell_indices)} cells.")
    else:
        # accept any cell with an anode and cathode
        accepted_cell_indices = np.where(
            ~df["Anode Type"].isna() &
            ~df["Cathode Type"].isna(),
        )[0]
        print(f"Accepted {len(accepted_cell_indices)} cells without checking N:P ratio.")

    # Re-write the Cell Number column to only include cells with both anode and cathode
    df["Cell Number"] = 0
    for cell_number, cell_index in enumerate(accepted_cell_indices):
        df.loc[cell_index, "Cell Number"] = cell_number + 1
        df.loc[cell_index, "Sample ID"] = f"{base_sample_id}_{cell_number + 1:02d}"


def main() -> None:
    """Full function to match cathodes with anodes and update the database.

    Read the cell assembly data from the database, calculate the capacity of the anodes and
    cathodes, and match the cathodes with the anodes to achieve the desired N:P ratio. Write the
    updated table back to the database.
    """
    sorting_method = int(sys.argv[1]) if len(sys.argv) >= 2 else 6

    print(f"Reading from database {DATABASE_FILEPATH}")
    print(f"Using sorting method {sorting_method}")

    # Connect to the database and create the Cell_Assembly_Table
    with sqlite3.connect(DATABASE_FILEPATH) as conn:
        # Read from database and calculate capacity
        df = pd.read_sql("SELECT * FROM Cell_Assembly_Table", conn)
        calculate_capacity(df)

        # Split the dataframe into sub-dataframes for each batch number
        batch_numbers = df["Batch Number"].unique()
        batch_numbers = batch_numbers[~np.isnan(batch_numbers)]

        for batch_number in batch_numbers:
            batch_mask = (
                (df["Batch Number"] == batch_number) &
                (df["Last Completed Step"] == 0) &
                (df["Error Code"] == 0) &
                (df["Anode Balancing Capacity (mAh)"] > 0) &
                (df["Cathode Balancing Capacity (mAh)"] > 0)
            )
            df_batch = df[batch_mask]
            # if no cells in this batch, skip
            if len(df_batch) == 0:
                print(f"Skipping batch number {batch_number} as there are no available cells.")
                continue
            row_indices = np.where(batch_mask)[0]
            n_rows = len(row_indices)
            n_rows_skipped = sum(df["Batch Number"] == batch_number) - n_rows
            print(f"Batch number {batch_number} has {n_rows} cells.")
            if n_rows_skipped:
                print(f"Ignoring {n_rows_skipped} cells that do not have "
                      f"Last Completed Step = 0 and Error Code = 0.")

            # Reorder the anode and cathode rack positions based on the sorting method
            match sorting_method:
                case 0: # Do not sort, do not check N:P ratio
                    anode_ind = np.arange(n_rows)
                    cathode_ind = np.arange(n_rows)
                    ratio_ind = np.arange(n_rows)

                case 1: # Do not sort
                    anode_ind = np.arange(n_rows)
                    cathode_ind = np.arange(n_rows)
                    ratio_ind = np.arange(n_rows)

                case 2: # Order by capacity
                    # I think this is always worse than the cost matrix approach
                    anode_sort = np.argsort(df_batch["Anode Balancing Capacity (mAh)"])
                    cathode_sort = np.argsort(df_batch["Cathode Balancing Capacity (mAh)"])
                    # Ensure that anode positions do not change
                    anode_ind = np.arange(n_rows)
                    cathode_ind = cathode_sort.iloc[np.argsort(anode_sort)]
                    ratio_ind = np.arange(n_rows)

                case 3: # Use cost matrix and linear sum assignment
                    anode_ind, cathode_ind = cost_matrix_assign(df_batch)
                    ratio_ind = np.arange(n_rows)

                case 4: # Use greedy 3D matching
                    anode_ind, cathode_ind, ratio_ind = cost_matrix_assign_3d(df_batch)

                case 5: # Use exact 3D matching
                    try:
                        anode_ind, cathode_ind, ratio_ind = cost_matrix_assign_3d(df_batch,exact=True)
                    except ValueError:
                        print("Exact matching took too long, using greedy matching instead")
                        anode_ind, cathode_ind, ratio_ind = cost_matrix_assign_3d(df_batch)

                case 6: # Choose automatically
                    # If all ratios are the same, use 2d matching
                    if (len(df_batch["Target N:P Ratio"].unique()) == 1 &
                        len(df_batch["Minimum N:P Ratio"].unique()) == 1 &
                        len(df_batch["Maximum N:P Ratio"].unique()) == 1):
                        anode_ind, cathode_ind = cost_matrix_assign(df_batch)
                        ratio_ind = np.arange(n_rows)
                    # Otherwise, try exact matching, if timeout use greedy matching
                    else:
                        try:
                            anode_ind, cathode_ind, ratio_ind = cost_matrix_assign_3d(df_batch,exact=True)
                        except ValueError:
                            print("Exact matching took too long, using greedy matching instead")
                            anode_ind, cathode_ind, ratio_ind = cost_matrix_assign_3d(df_batch)

            # Rearrange the electrodes in the main dataframe
            rearrange_electrode_columns(df, row_indices, anode_ind, cathode_ind, ratio_ind)

        # Read base_sample_id from the settings table
        df_settings = pd.read_sql("SELECT * FROM Settings_Table", conn)
        base_sample_id = df_settings.loc[df_settings["key"] == "Base Sample ID", "value"].to_numpy()[0]

        # Update the actual N:P ratio, accepted cell numbers and sample ID in the main dataframe
        if sorting_method == 0:
            update_cell_numbers(df, base_sample_id, check_NP_ratio=False)
        else:
            update_cell_numbers(df, base_sample_id)

        # Write the updated table back to the database
        df.to_sql("Cell_Assembly_Table", conn, index=False, if_exists="replace")
        print("Updated database successfully")

if __name__ == "__main__":
    main()
