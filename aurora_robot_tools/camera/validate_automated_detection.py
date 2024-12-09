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

from process_image import ProcessImages

#%%

compare_manual = True
compare_cathodes = True
modify = False
compare_modify = False
significant_digits = False

# Get Seaborn's default blue and red
sns_palette = sns.color_palette("deep")
seaborn_blue = sns_palette[0]  # First color in the palette (blue)
seaborn_red = sns_palette[3]   # Fourth color in the palette (red)

#%%

if compare_manual:
    folder = "C:/lisc_gen14/json"

    # Load the adjusted JSON file
    name_adj = "alignment_adjusted_gk.lisc_gen14.json" # "alignment_final_adjusted.lisc_gen14.json"
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
    with pd.ExcelWriter(os.path.join("C:/lisc_gen14/data", "data_manual_final_gk.xlsx")) as writer: # "C:/lisc_gen14/data", "data_manual_final.xlsx"
        df_adj.to_excel(writer, sheet_name='coordinates', index=False)

    # Load the automated JSON file
    data_dir = os.path.join(folder, "alignment_final.lisc_gen14.json")  # "alignment_final.lisc_gen14.json"
    with open(data_dir, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data["alignment"]) # Convert the "alignment" key into a DataFrame
    df["dz_mm_corr"] = np.sqrt(df["dx_mm_corr"]**2 + df["dy_mm_corr"]**2).round(3)
    print(df.head())

    # only look at one pressing tool:
    #p = 1
    #df_adj = df_adj[df_adj["press"] == p]
    #df = df[df["press"] == p]

    # Compute difference
    numerical_columns = ["r_mm", "dx_mm_corr", "dy_mm_corr", "dz_mm_corr"]
    if df.shape == df_adj.shape:
        # Compute the difference
        df_diff = df[numerical_columns] - df_adj[numerical_columns]
        df_diff = pd.concat([df_diff, df[["cell", "step", "press"]]], axis=1)
    else:
        raise ValueError("DataFrames df and df_adj must have the same shape")
    print(df_diff.head())
    df_diff = df_diff.dropna()
    df_diff["dz_mm_corr"] = np.sqrt((df["dx_mm_corr"] - df_adj["dx_mm_corr"])**2 -
                                    (df["dy_mm_corr"] - df_adj["dy_mm_corr"])**2)

    # df_diff.to_excel("C:/lisc_gen14/data/svfe_cathode_diff.xlsx")

    # Show differences in plot
    rows_to_plot = ["dx_mm_corr", "dy_mm_corr", "dz_mm_corr"]
    steps_to_plot = [0, 2, 6, 8]

    # Define colors for steps
    colors = cm.Set2.colors[:len(steps_to_plot)]
    colors = [seaborn_blue, seaborn_blue, seaborn_blue, seaborn_blue]

    # Create a figure with a 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=(12, 12), sharey=True)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Y-Axis labels
    y_labels = ["dx automatic\n- dx manual\n", "dy automatic\n- dy manual\n", "r automatic\n- r manual\n"]
    y_labels = ["$\Delta$x", "$\Delta$y", "$\Delta$r"]
    # Titles
    titles = {0: "Pressing Tool", 2: "Anode", 6: "Cathode", 8: "Spring"}

    # Loop through rows and steps to create the plots
    for row_idx, row in enumerate(rows_to_plot):
        for step_idx, step in enumerate(steps_to_plot):
            # Filter data for the current step
            step_data = df_diff[df_diff["step"] == step]

            # Select the current axis
            ax = axes[row_idx, step_idx]

            # Create a swarm plot
            sns.swarmplot(
                x=[""] * len(step_data[row]),  # Dummy x-axis as swarmplot requires a categorical x-axis
                y=step_data[row],
                ax=ax,
                color=colors[step_idx],
                alpha=1,
               size=4  # Adjust size of markers
            )
            # Create a violin plot with no inner part (only the colored frame)
            sns.violinplot(
                x=[""] * len(step_data[row]),  # Dummy x-axis as violinplot requires a categorical x-axis
                y=step_data[row],
                ax=ax,
                palette=["lightgray"],  # Use predefined colors
                alpha = 0.3,
                cut=0,  # Limit the density curve to the range of the data
                scale="width",  # Scale to match width across plots
                inner=None,  # Remove the inner distribution part (only show the frame)
                linewidth=2  # Optionally, adjust line width of the border
            )

            # Set the title and labels
            ax.set_title(f"{titles[step]}" if row_idx == 0 else "", fontsize=18)
            ax.set_ylabel(f"{y_labels[row_idx]} [mm]" if step_idx == 0 else "", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.grid(True, linestyle="--", alpha=0.6)

    axes[0, 0].text(-0.25,1.25,f"a)",fontsize=18,ha='left',va='top',transform=axes[0, 0].transAxes)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    #plt.show()

if compare_cathodes:

    gen15 = pd.read_excel("C:/lisc_gen14/data/svfe_cathode_diff.xlsx")

    # Create a figure with a 3x4 grid
    fig, axes = plt.subplots(3, 2, figsize=(6, 12), sharey=True)

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Y-Axis labels
    y_labels = ["dx automatic\n- dx manual\n", "dy automatic\n- dy manual\n", "r automatic\n- r manual\n"]
    y_labels = ["$\Delta$x", "$\Delta$y", "$\Delta$r"]
    # Titles
    titles = {0: "NMC 622", 1: "NMC 83"}

    # Loop through rows and steps to create the plots
    for row_idx, row in enumerate(rows_to_plot):
        for step_idx, step in enumerate([6, 6]):
            # Filter data for the current step
            if step_idx == 0:
                step_data = df_diff[df_diff["step"] == step]
            else:
                step_data = gen15[gen15["step"] == step]

            # Select the current axis
            ax = axes[row_idx, step_idx]

            # Create a swarm plot
            sns.swarmplot(
                x=[""] * len(step_data[row]),  # Dummy x-axis as swarmplot requires a categorical x-axis
                y=step_data[row],
                ax=ax,
                color=colors[step_idx],
                alpha=1,
               size=4  # Adjust size of markers
            )
            # Create a violin plot with no inner part (only the colored frame)
            sns.violinplot(
                x=[""] * len(step_data[row]),  # Dummy x-axis as violinplot requires a categorical x-axis
                y=step_data[row],
                ax=ax,
                palette=["lightgray"],  # Use predefined colors
                alpha = 0.3,
                cut=0,  # Limit the density curve to the range of the data
                scale="width",  # Scale to match width across plots
                inner=None,  # Remove the inner distribution part (only show the frame)
                linewidth=2  # Optionally, adjust line width of the border
            )

            # Set the title and labels
            ax.set_title(f"{titles[step_idx]}" if row_idx == 0 else "", fontsize=18)
            ax.set_ylabel(f"{y_labels[row_idx]} [mm]" if step_idx == 0 else "", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.set_xticks([])
            ax.set_xlabel('')
            ax.grid(True, linestyle="--", alpha=0.6)

    axes[0, 0].text(-0.25,1.25,f"b)",fontsize=18,ha='left',va='top',transform=axes[0, 0].transAxes) 

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

#%%

if modify:
    # Generate modified images to check robustness
    modification = "crop"  # Change this to "warping", "extremewarping", "smallwarping", "blur", or "crop"
    # Input and output directories
    input_folder = r"C:\241127_modified_images\raw"
    output_folder = "C:/241127_modified_images/crop"

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
                crop_cols = cols * 11 // 12  # Retain only 2/3 of the width
                modified_image = content[:, :crop_cols]
            else:
                modified_image = content
                print("not modified")

            # Save the warped image to a new .h5 file
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('image', data=modified_image)

            # Optional: Display the modified image
            plt.imshow(modified_image, cmap='gray')
            plt.title(f"Modified Image ({modification}): {filename}")
            plt.show()

#%%

if compare_modify:
    # Evaluate the robustness of the circle detection with the modified images
    df_ref = pd.read_excel("C:/241127_modified_images/original/data/data.xlsx")
    df1 = pd.read_excel("C:/241127_modified_images/smallwarping/data/data.xlsx")
    df2 = pd.read_excel("C:/241127_modified_images/blur/data/data.xlsx")
    df3 = pd.read_excel("C:/241127_modified_images/crop/data/data.xlsx")

    # Calculate deviations
    deviation_df1 = df1[["dx_mm", "dy_mm"]] - df_ref[["dx_mm", "dy_mm"]]
    deviation_df2 = df2[["dx_mm", "dy_mm"]] - df_ref[["dx_mm", "dy_mm"]]
    deviation_df3 = df3[["dx_mm", "dy_mm"]] - df_ref[["dx_mm", "dy_mm"]]

    # drop nan
    deviation_df1 = deviation_df1.dropna()
    deviation_df2 = deviation_df2.dropna()
    deviation_df3 = deviation_df3.dropna()

    # Define the columns to consider
    columns = ["dx_mm", "dy_mm"]

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="tight")

    for i, col in enumerate(columns):
        ax = axes[i]

        # Prepare data for swarm and violin plots
        data = [deviation_df1[col], deviation_df2[col], deviation_df3[col]]

        # Create a swarm plot on top of the violin plot
        sns.swarmplot(
            x=["warping"] * len(deviation_df1[col]),  # Dummy x-axis for each dataset
            y=deviation_df1[col], ax=ax,
            color=seaborn_blue,  # Color for the points in the swarm plot
            alpha=1,  # Fully opaque
            size=4  # Adjust size of markers
        )

        sns.swarmplot(
            x=["blur"] * len(deviation_df2[col]),  # Dummy x-axis for each dataset
            y=deviation_df2[col], ax=ax,
            color=seaborn_blue,  # Color for the points in the swarm plot
            alpha=1,
            size=4
        )

        sns.swarmplot(
            x=["crop"] * len(deviation_df3[col]),  # Dummy x-axis for each dataset
            y=deviation_df3[col], ax=ax,
            color=seaborn_blue,  # Color for the points in the swarm plot
            alpha=1,
            size=4
        )

        ax.set_title(f"Deviation in {col}")
        if i == 0:
            ax.set_ylabel("Deviation [mm]")
        ax.set_xlabel("Method")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Calculate and annotate standard deviation
        std_devs = [np.std(deviation_df1[col]), np.std(deviation_df2[col]), np.std(deviation_df3[col])]
        for j, std in enumerate(std_devs):
            ax.text(j, max(data[j]) + 0.1, f"σ={std:.2f}",
                    ha='center', va='bottom', fontsize=10, color='gray')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Combine the datasets into one DataFrame for easy plotting
    deviation_df1['method'] = 'warping'
    deviation_df2['method'] = 'blur'
    deviation_df3['method'] = 'crop'

    df_combined = pd.concat([deviation_df1, deviation_df2, deviation_df3])

    # Create the subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, layout="tight")

    for i, col in enumerate(["dx_mm", "dy_mm"]):  # Adjust columns as per your data
        ax = axes[i]

        # Create a catplot with kind="violin" (this will show a violin shape around each swarm)
        sns.catplot(
            data=df_combined, 
            x="method", 
            y=col, 
            kind="violin", 
            color=".9",  # Light color for the violins
            inner=None,  # No inner distribution part (only the border)
            ax=ax  # Specify the axis to plot on
        )

        # Overlay the swarm plot on top of the violin plot
        sns.swarmplot(
            x="method", 
            y=col, 
            data=df_combined, 
            ax=ax, 
            color="black",  # Color for the points
            alpha=1,  # Fully opaque
            size=4  # Adjust size of markers
        )

        ax.set_title(f"Deviation in {col}")
        if i == 0:
            ax.set_ylabel("Deviation [mm]")
        ax.set_xlabel("Method")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Calculate and annotate standard deviation
        std_devs = [np.std(deviation_df1[col]), np.std(deviation_df2[col]), np.std(deviation_df3[col])]
        for j, std in enumerate(std_devs):
            ax.text(j, max([deviation_df1[col].max(), deviation_df2[col].max(), deviation_df3[col].max()]) + 0.1,
                    f"σ={std:.2f}",
                    ha='center', va='bottom', fontsize=10, color='gray')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# %%

if significant_digits:

    # import circle detection function:
    from process_image import _detect_circles

    # conversion factor
    mm_to_pixel = 10
    # path
    path = "C:/lisc_gen14/significant_digits/images"
    # no thickness correction needed since it is step 0

    # Get images
    image_dict = {}
    # Iterate through all files in the folder
    for filename in os.listdir(path):
        if filename.lower().endswith('.jpg'):  # Check if file is a .jpg
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)  # Read image as an array
            if image is not None:
                image_dict[filename] = image  # Add to dictionary

    # loop over pressing tool positions
    pressing_tools = [1, 2, 3, 4, 5, 6]
    # create data frame with values
    circles_df = pd.DataFrame()
    for tool in pressing_tools:
        # store values in dict & data frame:
        name_list = []
        x_mm_list = []
        y_mm_list = []
        circles_dict = {}
        # loop over dict to find pressing tool
        for key, value in image_dict.items():
            p = int(key.split(".")[0].split("_")[-1])
            if p == tool:
                # convert to 8-Bit
                value = value / np.max(value) * 255
                img = value.astype(np.uint8)
                # process image before detection
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.GaussianBlur(img, (5, 5), 2)
                # detect circle
                center, rad, image_with_circles = _detect_circles(img,
                                                                (int(9.75*mm_to_pixel), int(10.25*mm_to_pixel)),
                                                                (30, 50))
                # convert to mm
                center_mm = [tuple(ti/mm_to_pixel for ti in c) for c in center][0]
                # x & y
                x_mm = center_mm[0]
                y_mm = center_mm[1]
                # save center in dictionary
                circles_dict[key.split(".")[0]] = center_mm
                # row for data frame
                name_list.append(key.split(".")[0])
                x_mm_list.append(x_mm)
                y_mm_list.append(y_mm)
                # save image
                cv2.imwrite(f'C:/lisc_gen14/significant_digits/detected_images/{key}', image_with_circles)

        # save center in data frame
        # circles_df[f"name {tool}"] = name_list
        circles_df[f"x{tool} [mm]"] = x_mm_list
        circles_df[f"y{tool} [mm]"] = y_mm_list

    print(circles_df)

    # Run full image (not only image sections)

    # PARAMETER
    folderpath = "G:/Limit/Lina Scholz/Images Camera Adjustment/11 Same Image Different Light"
    # CLASS & FUNCTIONS
    obj = ProcessImages(folderpath)
    list = obj.load_files()
    df = obj.store_data(list)
    df = obj.get_centers(df)
    df = obj.correct_for_thickness(df)
    coordinates_df = obj.save()
