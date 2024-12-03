""" Lina Scholz

Script to compare manually improved detection with only automated detection.

"""

import os
import cv2
import json
import h5py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#%%

compare_manual = True
modify = False
compare_modify = False

#%%

if compare_manual:
    folder = "C:/241105_svfe_gen15/json"

    # Load the adjusted JSON file
    name_adj = "alignment_adjusted.241105_svfe_gen15.json"
    data_dir_adj = os.path.join(folder, name_adj)
    with open(data_dir_adj, 'r') as file:
        data_adj = json.load(file)
    df_adj = pd.DataFrame(data_adj["alignment"]) # Convert the "alignment" key into a DataFrame
    df_adj["dz_mm_corr"] = np.sqrt(df_adj["dx_mm_corr"]**2 + df_adj["dy_mm_corr"]**2).round(3)
    print(df_adj.head())
    # Save back as data frame
    name_list = name_adj.split(".")
    name_list.pop()
    name_save = ".".join(map(str, name_list))
    with pd.ExcelWriter(os.path.join("C:/241105_svfe_gen15/data", "data_manual.xlsx")) as writer:
        df_adj.to_excel(writer, sheet_name='coordinates', index=False)

    # Load the automated JSON file
    data_dir = os.path.join(folder, "alignment.241105_svfe_gen15.json")
    with open(data_dir, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data["alignment"]) # Convert the "alignment" key into a DataFrame
    df["dz_mm_corr"] = np.sqrt(df["dx_mm_corr"]**2 + df["dy_mm_corr"]**2).round(3)
    print(df.head())

    # only look at one pressing tool:
    # p = 6
    # df_adj = df_adj[df_adj["press"] == p]
    # df = df[df["press"] == p]

    # Compute difference
    df_adj = df_adj.dropna()
    df = df.dropna()
    numerical_columns = ["r_mm", "dx_mm_corr", "dy_mm_corr", "dz_mm_corr"]
    if df.shape == df_adj.shape:
        # Compute the difference
        df_diff = df[numerical_columns] - df_adj[numerical_columns]
        df_diff = pd.concat([df_diff, df[["cell", "step", "press"]]], axis=1)
    else:
        raise ValueError("DataFrames df and df_adj must have the same shape")
    print(df_diff.head())
    df_diff = df_diff.dropna()

    # Show differences in plot
    rows_to_plot = ["dx_mm_corr", "dy_mm_corr", "dz_mm_corr"]
    steps_to_plot = [0, 2, 6, 8]

    # Define colors for steps
    colors = cm.Set2.colors[:len(steps_to_plot)]
    colors = ["lightgrey", "lightgrey", "lightgrey", "lightgrey"]

    # Create a figure with a 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 12), sharey=True)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Y-Axis labels
    y_labels = ["dx automatic\n- dx manual\n", "dy automatic\n- dy manual\n", "r automatic\n- r manual\n"]
    # Titles
    titles = {0: "Pressing Tool", 2: "Anode", 6: "Cathode", 8: "Spring"}

    # Loop through rows and steps to create the plots
    for row_idx, row in enumerate(rows_to_plot):
        for step_idx, step in enumerate(steps_to_plot):
            # Filter data for the current step
            step_data = df_diff[df_diff["step"] == step]

            # Select the current axis
            ax = axes[row_idx, step_idx]

            # Create a boxplot
            ax.boxplot(step_data[row], patch_artist=True, boxprops=dict(facecolor=colors[step_idx], color='black'),
                    medianprops=dict(color='red'), whiskerprops=dict(color='black'))

            # Remove extra space around the boxplot
            ax.set_xlim(0.9, 1.1)  # Shrinks x-axis to tightly fit the boxplot
            ax.margins(x=0)

            # Set the title and labels
            ax.set_title(f"{titles[step]}" if row_idx == 0 else "", fontsize=18)
            ax.set_ylabel(f"{y_labels[row_idx]} [mm]" if step_idx == 0 else "", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.grid(True, linestyle="--", alpha=0.6)

    # Adjust layout
    # fig.suptitle("Deviations: Automated vs. Manual Circle Detection (by Step)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#%%

if modify:
    # Generate modified images to check robustness
    modification = "original"  # Change this to "warping", "extremewarping", "smallwarping", "blur", or "crop"
    # Input and output directories
    input_folder = r"C:\241127_modified_images\raw"
    output_folder = "C:/241127_modified_images/original"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all .h5 files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".h5"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Process the file
            with h5py.File(input_path, 'r') as f:
                content = f['image'][:]

            if modification == "warping":
                # Apply a warp transformation
                rows, cols = content.shape[:2]
                src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                dst_points = np.float32([
                    [cols * 0.03, rows * 0.01],    # Slightly move the top-left corner
                    [cols-1 - cols * 0.01, rows * 0.03],  # Slightly move the top-right corner
                    [cols * 0.01, rows-1 - rows * 0.04]   # Slightly move the bottom-left corner
                ])
                matrix = cv2.getAffineTransform(src_points, dst_points)
                modified_image = cv2.warpAffine(content, matrix, (cols, rows))
            elif modification == "extremewarping":
                # Apply a warp transformation
                rows, cols = content.shape[:2]
                src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                dst_points = np.float32([
                    [cols * 0.05, rows * 0.02],    # Slightly move the top-left corner
                    [cols-1 - cols * 0.01, rows * 0.06],  # Slightly move the top-right corner
                    [cols * 0.01, rows-1 - rows * 0.07]   # Slightly move the bottom-left corner
                ])
                matrix = cv2.getAffineTransform(src_points, dst_points)
                modified_image = cv2.warpAffine(content, matrix, (cols, rows))
            elif modification == "smallwarping":
                # Apply a warp transformation
                rows, cols = content.shape[:2]
                src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1]])
                dst_points = np.float32([
                    [cols * 0.02, rows * 0.01],    # Slightly move the top-left corner
                    [cols-1 - cols * 0.02, rows * 0.01],  # Slightly move the top-right corner
                    [cols * 0.01, rows-1 - rows * 0.02]   # Slightly move the bottom-left corner
                ])
                matrix = cv2.getAffineTransform(src_points, dst_points)
                modified_image = cv2.warpAffine(content, matrix, (cols, rows))
            elif modification == "blur":
                # Apply Gaussian blur to the image
                modified_image = cv2.GaussianBlur(content, (5, 5), 0)
            elif modification == "crop":
                # Crop 1/3 of the image from the right
                rows, cols = content.shape[:2]
                crop_cols = cols * 2 // 3  # Retain only 2/3 of the width
                modified_image = content[:, :crop_cols]
            else:
                modified_image = content
                print("not modified")

            # Save the warped image to a new .h5 file
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('image', data=modified_image)

            # Optional: Display the modified image
            # plt.imshow(modified_image, cmap='gray')
            # plt.title(f"Modified Image ({modification}): {filename}")
            # plt.show()

#%%

if compare_modify:
    # Evaluate the robustness of the circle detection with the modified images
    df_ref = pd.read_excel("C:/241127_modified_images/original/data/data.xlsx")
    df1 = pd.read_excel("C:/241127_modified_images/smallwarping/data/data.xlsx")
    df2 = pd.read_excel("C:/241127_modified_images/crop/data/data.xlsx")
    df3 = pd.read_excel("C:/241127_modified_images/blur/data/data.xlsx")

    # List of DataFrames and variable names to compare
    data_frames = [df1, df2, df3]
    variables = ['r_mm', 'x', 'y']

    # Dictionary to store deviations for each DataFrame
    deviation_summary = {}

    # Calculate deviations
    for i, df in enumerate(data_frames, start=1):
        deviations = {}
        for var in variables:
            # Compute absolute deviations
            deviations[var] = np.abs(df[var] - df_ref[var])
        deviation_summary[f"data{i}"] = deviations

    # Summarize deviations numerically
    summary_table = {}
    for name, devs in deviation_summary.items():
        summary_table[name] = {var: dev.mean() for var, dev in devs.items()}

    # Convert summary to a DataFrame for visualization
    summary_df = pd.DataFrame(summary_table).T
    print("Deviation Summary (Mean Absolute Deviations):")
    print(summary_df)

    # Plot deviations for each variable
    plt.figure(figsize=(14, 10))
    for i, var in enumerate(variables, start=1):
        plt.subplot(2, 2, i)
        sns.boxplot(data=[deviation_summary[f"data{j}"][var] for j in range(1, 4)])  # Adjusted to range(1, 4)
        plt.xticks(ticks=range(3), labels=[f"data{j}" for j in range(1, 4)])  # Adjusted to 3 labels
        plt.title(f"Deviations of {var}")
        plt.ylabel("Deviation")
    plt.tight_layout()
    plt.show()

    print(deviation_summary)

# %%
