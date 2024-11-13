""" Lina Scholz

Script to read in images from folder, transform warped rectangle in straight rectangle,
detect centers of all parts.
"""

import math
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%% CLASS

class Alignment:
    def __init__(self, path, graham=False):
        self.path = path # path to images
        self.df = pd.read_excel(os.path.join(path, "data/data.xlsx"), sheet_name="coordinates")
        self.alignment_df = pd.DataFrame()
        self.mm_to_pixel = 10
        self.selected_steps = [0, 1, 2, 4, 6, 7, 8, 9]

        # ------------------------------------------------------------------------------------------
        if graham: # Grahams cells
            base_string = "241004_kigr_gen5_"
            # Create a list with formatted strings
            self.sample_ID = [f"{base_string}{str(i).zfill(2)}" for i in range(1, 37)]

        else: # Linas cells
            base_string = "241022_lisc_gen14_"
            self.sample_ID = [
            '241022_lisc_gen14_2-13_02', '241022_lisc_gen14_2-13_03', '241022_lisc_gen14_2-13_04',
            '241022_lisc_gen14_2-13_05', '241022_lisc_gen14_2-13_06', '241022_lisc_gen14_2-13_07',
            '241022_lisc_gen14_2-13_08', '241022_lisc_gen14_2-13_09', '241022_lisc_gen14_2-13_10',
            '241022_lisc_gen14_2-13_11', '241022_lisc_gen14_2-13_12', '241022_lisc_gen14_2-13_13',
            '241022_lisc_gen14_14_36_14', '241022_lisc_gen14_14_36_15', '241022_lisc_gen14_14_36_16',
            '241022_lisc_gen14_14_36_17', '241022_lisc_gen14_14_36_18', '241022_lisc_gen14_14_36_19',
            '241022_lisc_gen14_14_36_20', '241022_lisc_gen14_14_36_21', '241022_lisc_gen14_14_36_22',
            '241022_lisc_gen14_14_36_23', '241022_lisc_gen14_14_36_24', '241022_lisc_gen14_14_36_25',
            '241022_lisc_gen14_14_36_26', '241022_lisc_gen14_14_36_27', '241022_lisc_gen14_14_36_28',
            '241022_lisc_gen14_14_36_29', '241022_lisc_gen14_14_36_30', '241022_lisc_gen14_14_36_31',
            '241022_lisc_gen14_14_36_32', '241022_lisc_gen14_14_36_33', '241022_lisc_gen14_14_36_34',
            '241022_lisc_gen14_14_36_35', '241022_lisc_gen14_14_36_36'
            ]

    def _intersection_area(self, d: float) -> float:
        """ Function to return percentage of area of intersection of the cathode.

        Args:
            d (float): distance between centers in mm

        Returns:
            perentage_area (float): percentage of cathode overlapping with anode
        """
        # radius in mm
        R1 = 15
        R2 = 14
        if d >= R1 + R2:
            area = 0 # No overlap
        elif d <= abs(R1 - R2) :
            area = math.pi * min(R1, R2) ** 2  # The smaller circle is fully contained
        else :
            part1 = R1**2 * math.acos((d**2 + R1**2 - R2**2) / (2 * d * R1))
            part2 = R2**2 * math.acos((d**2 + R2**2 - R1**2) / (2 * d * R2))
            part3 = 0.5 * math.sqrt((-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2))
            area = part1 + part2 - part3
        percentage_area = area / (math.pi * R2**2) * 100 # area of cathode overlapping with anode
        return percentage_area

    def plot_coordinates_by_cell(self):
        # Get unique cell numbers
        unique_cells = self.df['cell'].unique()
        radii_dict = {0: 20, 1: 20, 2: 15, 4: 16, 5: 16, 6: 14, 7: 19, 8: 16, 9: 20}
        colors = plt.cm.get_cmap('Blues')
        color_range = np.linspace(0.4, 1, len(self.selected_steps))
        color_range = color_range[::-1]
        step_name = {0: "press", 1: "bottom", 2: "anode", 4: "separator",
                     6: "cathode", 7: "spacer", 8: "spring", 9: "top"}

        # Loop over each cell and create a plot
        for cell in unique_cells:
            # Filter DataFrame for the current cell and selected steps
            cell_data = self.df[(self.df['cell'] == cell) & (self.df['step'].isin(self.selected_steps))]
            # Get the reference point (0,0) for `step 0`
            ref_x, ref_y = cell_data[cell_data['step'] == 0][['x', 'y']].values[0]
            # Adjust coordinates relative to the step 0 reference point
            cell_data.loc[:, 'x'] = (cell_data.loc[:, 'x'] - ref_x) / self.mm_to_pixel
            cell_data.loc[:, 'y'] = (cell_data.loc[:, 'y'] - ref_y) / self.mm_to_pixel

            cell_data["x"]

            # Create a figure and axis for each cell plot
            fig, ax = plt.subplots(figsize=(10, 10), layout="tight")
            # Loop through each selected step and plot the points with circles
            for i, (step, row) in enumerate(cell_data.groupby('step')):
                color = colors(color_range[i])

                # Get coordinates and radius
                x, y = row['x'].values[0], row['y'].values[0]
                radius = radii_dict.get(step, 200)  # Default to 200 if step not in dictionary
                # Plot the center point
                ax.scatter(x, y, color=color, label=f'{step_name[step]}', s=50, zorder=5)
                # Add a circle around the center point
                circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=1.5)
                ax.add_patch(circle)

                # Annotate with step label, offset from center
                ax.annotate(
                    f"{step_name[step]}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(10, 10),  # Offset the label
                    ha='center',
                    fontsize=9,
                    color=color
                )

            # Add titles, labels, and legend
            ax.set_title(f"Coordinates for Cell {cell}", fontsize=12)
            ax.set_xlabel("x [mm]", fontsize=10)
            ax.set_ylabel("y [mm]", fontsize=10)
            ax.legend(loc="upper left", title="Steps", fontsize=8, title_fontsize=9)
            ax.set_aspect('equal')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            ax.grid(True)

            # Save the plot as a JPG file named by the cell number
            data_dir = os.path.join(self.path, "plot")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            plt.savefig(os.path.join(data_dir, f"center_cell_{cell}.jpg"), format="jpg")
            plt.close()

    def plot_differences(self, step, name):
        # create folder plot if not existent
        if not os.path.exists('plot'):
            os.makedirs('plot')

        df_filtered = self.df[self.df['step'].isin([0, 2])] # filter for specific steps
        # pivot data frame to have step 0 and 2 next to each other
        df_pivot = df_filtered.pivot_table(index='cell', columns='step', values=['x', 'y'])
        # calculate difference
        df_pivot['x_diff'] = (df_pivot['x', step] - df_pivot['x', 0]) / self.mm_to_pixel
        df_pivot['y_diff'] = (df_pivot['y', step] - df_pivot['y', 0]) / self.mm_to_pixel

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df_pivot.index, df_pivot['x_diff'], color='blue', label='x_diff', s=50)
        ax.scatter(df_pivot.index, df_pivot['y_diff'], color='red', label='y_diff', s=50)

        ax.set_xlabel('Cell Number / Rack Position')
        ax.set_ylabel('Misalignment [mm]')
        ax.set_title(f'{name} Misalignment')
        ax.legend()
        ax.grid(True)

        # Save the plot as a JPG file named by the cell number
        data_dir = os.path.join(self.path, "plot")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        plt.savefig(os.path.join(data_dir, 'differences_plot_anode.png'), format="jpg")
        plt.close()

    def compute_differences(self):
        # Filtern der Daten für die ausgewählten steps
        df_filtered = self.df[self.df['step'].isin(self.selected_steps)]
        df_filtered = df_filtered[["cell", "step", "x", "y"]]

        # pivot data frame to have all selected steps next to each other
        df_pivot_x = df_filtered.pivot_table(index='cell', columns='step', values='x')
        df_pivot_y = df_filtered.pivot_table(index='cell', columns='step', values='y')

        # calculate differences for each step compared to step 0
        diff_x = (df_pivot_x[0] - df_pivot_x) / self.mm_to_pixel # in mm
        diff_y = (df_pivot_y[0] - df_pivot_y) / self.mm_to_pixel # in mm
        diff_df = pd.concat([diff_x, diff_y], axis=1) # combine
        # Rename columns
        diff_df.columns = [f"x_diff_{step}" if i < len(self.selected_steps) - 1 else f"y_diff_{step}"
                        for i, step in enumerate(self.selected_steps[1:]*2)]
        return diff_df

    def plot_correlation_matrix(self, df):
        # Berechnen der Korrelationen zwischen den Differenzen der Koordinaten
        correlation_matrix = df.corr()

        # Plot der Korrelationsmatrix als Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
        plt.title("Korrelationsmatrix der Differenzen zwischen den Koordinaten")
        plt.show()

#%% RUN CODE
if __name__ == '__main__':

    # PARAMETER
    folderpath = "C:/lisc_gen14"

    obj = Alignment(folderpath)
    obj.plot_coordinates_by_cell()
    obj.plot_differences(step=2, name="Anode")
    differences_matrix = obj.compute_differences()
    obj.plot_correlation_matrix(differences_matrix)

