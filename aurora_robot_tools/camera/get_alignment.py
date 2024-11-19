""" Lina Scholz

Script to analyse center coordinates and alignment of different parts from the data of proces_images.py.
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
        self.alignment_df = pd.DataFrame(columns=["cell", "press", "z_electrodes", "intersection_area",
                                                  "x1", "y1", "z1", "x2", "y2", "z2", "x4", "y4", "z4",
                                                  "x6", "y6", "z6", "x7", "y7", "z7", "x8", "y8", "z8", 
                                                  "x9", "y9", "z9"])
        self.mm_to_pixel = 10
        self.selected_steps = [0, 1, 2, 4, 6, 7, 8, 9]
        self.unique_cells = self.df['cell'].unique()

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
        elif d <= abs(R1 - R2)/2:
            area = math.pi * min(R1, R2) ** 2  # The smaller circle is fully contained
        else:
            # Compute alpha and beta, clamping the values for acos to prevent domain errors
            # https://www.geeksforgeeks.org/area-of-intersection-of-two-circles/
            alpha = math.acos(max(-1, min(1, ((R1**2 + d**2 - R2**2) / (2 * d * R1))))) * 2
            beta = math.acos(max(-1, min(1, ((R2**2 + d**2 - R1**2) / (2 * d * R2))))) * 2
            a1 = (0.5 * beta * R2 * R2 ) - (0.5 * R2 * R2 * math.sin(beta))
            a2 = (0.5 * alpha * R1 * R1) - (0.5 * R1 * R1 * math.sin(alpha))
            area = math.floor(a1 + a2)
        percentage_area = round(area / (math.pi * R2**2) * 100, 2) # area of cathode overlapping with anode
        return percentage_area

    def plot_coordinates_by_cell(self, draw_circle=False) -> pd.DataFrame:
        """ Plots the center points of all parts for each cell.

        The circle of the size of the parts is only plotted if desired. The images with the center
        points of each part are saved as well as the data frame with the distances of the differnt
        parts to the pressing tool reference coordinate. The pressing tool coordinate is set to zero
        as reference.

        Args:
            draw_circle (bool): whether to draw the circle for the size of the part

        Returns:
            self.alignment_df (data frame): contains all x, y, z coordinates of each part with the
                                            pressing tool as reference coordinate
        """
        # Get unique cell numbers
        radii_dict = {0: 20, 1: 20, 2: 15, 4: 16, 5: 16, 6: 14, 7: 19, 8: 16, 9: 20}
        colors = plt.cm.get_cmap('Blues')
        color_range = np.linspace(0.4, 1, len(self.selected_steps))
        color_range = color_range[::-1]
        step_name = {0: "press", 1: "bottom", 2: "anode", 4: "separator",
                     6: "cathode", 7: "spacer", 8: "spring", 9: "top"}

        # Loop over each cell and create a plot
        for cell in self.unique_cells:
            # Filter DataFrame for the current cell and selected steps
            cell_df = self.df[(self.df['cell'] == cell) & (self.df['step'].isin(self.selected_steps))]
            # Get the reference point (0,0) for `step 0`
            ref_x, ref_y = cell_df[cell_df['step'] == 0][['x', 'y']].values[0]
            # Adjust coordinates relative to the step 0 reference point
            cell_df.loc[:, 'x'] = (cell_df.loc[:, 'x'] - ref_x) / self.mm_to_pixel
            cell_df.loc[:, 'y'] = (cell_df.loc[:, 'y'] - ref_y) / self.mm_to_pixel
            cell_df["z"] = np.sqrt(cell_df["x"]**2 + cell_df["y"]**2).round(3)
            x_electrodes = (float(cell_df.loc[cell_df['step']==6,'x'].values -
                                  cell_df.loc[cell_df['step']==2,'x'].values))
            y_electrodes = (float(cell_df.loc[cell_df['step']==6,'x'].values -
                                  cell_df.loc[cell_df['step']==2,'x'].values))
            z_electrodes = round(math.sqrt(x_electrodes**2 + y_electrodes**2), 3)

            # store all alignments with respect to pressing tool
            row = {"cell": cell, "press": int(cell_df["press"].values[0]),
                   "z_electrodes": z_electrodes,
                   "intersection_area": self._intersection_area(z_electrodes),
                   "x1": float(cell_df.loc[cell_df['step'] == 1, 'x']),
                   "y1": float(cell_df.loc[cell_df['step'] == 1, 'y']),
                   "z1": float(cell_df.loc[cell_df['step'] == 1, 'z']),
                   "x2": float(cell_df.loc[cell_df['step'] == 2, 'x']),
                   "y2": float(cell_df.loc[cell_df['step'] == 2, 'y']),
                   "z2": float(cell_df.loc[cell_df['step'] == 2, 'z']),
                   "x4": float(cell_df.loc[cell_df['step'] == 4, 'x']),
                   "y4": float(cell_df.loc[cell_df['step'] == 4, 'y']),
                   "z4": float(cell_df.loc[cell_df['step'] == 4, 'z']),
                   "x6": float(cell_df.loc[cell_df['step'] == 6, 'x']),
                   "y6": float(cell_df.loc[cell_df['step'] == 6, 'y']),
                   "z6": float(cell_df.loc[cell_df['step'] == 6, 'z']),
                   "x7": float(cell_df.loc[cell_df['step'] == 7, 'x']),
                   "y7": float(cell_df.loc[cell_df['step'] == 7, 'y']),
                   "z7": float(cell_df.loc[cell_df['step'] == 7, 'z']),
                   "x8": float(cell_df.loc[cell_df['step'] == 8, 'x']),
                   "y8": float(cell_df.loc[cell_df['step'] == 8, 'y']),
                   "z8": float(cell_df.loc[cell_df['step'] == 8, 'z']),
                   "x9": float(cell_df.loc[cell_df['step'] == 9, 'x']),
                   "y9": float(cell_df.loc[cell_df['step'] == 9, 'y']),
                   "z9": float(cell_df.loc[cell_df['step'] == 9, 'z'])}
            self.alignment_df.loc[len(self.alignment_df)] = row

            # Create a figure and axis for each cell plot
            fig, ax = plt.subplots(figsize=(10, 10), layout="tight")
            # Loop through each selected step and plot the points with circles
            for i, (step, row) in enumerate(cell_df.groupby('step')):
                color = colors(color_range[i])

                # Get coordinates and radius
                x, y = row['x'].values[0], row['y'].values[0]
                radius = radii_dict.get(step, 200)  # Default to 200 if step not in dictionary
                # Plot the center point
                ax.scatter(x, -y, color=color, label=f'{step_name[step]}', s=50, zorder=5)
                if draw_circle:
                    # Add a circle around the center point
                    circle = plt.Circle((x, y), radius, color=color, fill=False, linewidth=1.5)
                    ax.add_patch(circle)

                # Annotate with step label, offset from center
                ax.annotate(
                    f"{step_name[step]}",
                    (x, -y),
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
            if draw_circle: # need larger range to see circle
                ax.set_xlim([-25, 25])
                ax.set_ylim([-25, 25])
            else:
                ax.set_xlim([-3, 3])
                ax.set_ylim([-3, 3])
            ax.grid(True)

            # Save the plot as a JPG file named by the cell number
            data_dir = os.path.join(self.path, "plot")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            plt.savefig(os.path.join(data_dir, f"center_cell_{cell}.jpg"), format="jpg")
            plt.close()

        # Save alignment data frame
        self.alignment_df = self.alignment_df.sort_values(by='cell')
        data_dir = os.path.join(self.path, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        with pd.ExcelWriter(os.path.join(data_dir, "alignment.xlsx")) as writer:
            self.alignment_df.to_excel(writer, sheet_name='alignment', index=False)
        self.alignment_df.to_csv(os.path.join(data_dir, "alignment.csv"), index=False)

        return self.alignment_df

    def plot_differences(self, step: int, name: str):
        """ Gives a plot of the part defined in the step and its alignment vs. cell number/rack position.

        Args:
            step (int): step of robot of part which should be investigated
            name (str): name of part
        """
        # create folder plot if not existent
        if not os.path.exists('plot'):
            os.makedirs('plot')

        df_filtered = self.df[self.df['step'].isin([0, step])] # filter for specific steps
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
        ax.set_ylim([-2.5, 2.5])
        ax.legend()
        ax.grid(True)

        # Save the plot as a JPG file named by the cell number
        data_dir = os.path.join(self.path, "plot")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        plt.savefig(os.path.join(data_dir, f'differences_plot_{name}.png'), format="jpg")
        plt.close()

#%% RUN CODE
if __name__ == '__main__':

    # PARAMETER
    folderpath = "C:/lisc_gen14"

    obj = Alignment(folderpath)
    obj.plot_coordinates_by_cell()
    obj.plot_differences(step=2, name="Anode")
    obj.plot_differences(step=6, name="Cathode")
    obj.plot_differences(step=7, name="Spacer")
    obj.plot_differences(step=8, name="Spring")



