""" Lina Scholz

Script to alanyse performance data with alignment data.
"""

import math
import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

#%% Append PERFORMACNE DATA

keys = ['Sample ID', 'Cycle', 'Charge capacity (mAh)', 'Discharge capacity (mAh)', 'Efficiency (%)',
        'Specific charge capacity (mAh/g)', 'Specific discharge capacity (mAh/g)', 'Normalised discharge capacity (%)',
        'Normalised discharge energy (%)', 'Charge average voltage (V)', 'Discharge average voltage (V)', 'Delta V (V)',
        'Charge average current (A)', 'Discharge average current (A)', 'Charge energy (mWh)', 'Discharge energy (mWh)',
        'Max voltage (V)', 'Formation C', 'Cycle C', 'Actual N:P ratio', 'Anode type', 'Cathode type',
        'Anode active material mass (mg)', 'Cathode active material mass (mg)', 'Electrolyte name',
        'Electrolyte description', 'Electrolyte amount (uL)', 'Rack position', 'First formation efficiency (%)',
        'First formation specific discharge capacity (mAh/g)', 'Initial specific discharge capacity (mAh/g)',
        'Initial efficiency (%)', 'Capacity loss (%)', 'Last specific discharge capacity (mAh/g)',
        'Last efficiency (%)', 'Formation average voltage (V)', 'Formation average current (A)', 'Initial delta V (V)',
        'Cycles to 95% capacity', 'Cycles to 90% capacity', 'Cycles to 85% capacity', 'Cycles to 80% capacity',
        'Cycles to 75% capacity', 'Cycles to 70% capacity', 'Cycles to 60% capacity', 'Cycles to 50% capacity',
        'Cycles to 95% energy', 'Cycles to 90% energy', 'Cycles to 85% energy', 'Cycles to 80% energy',
        'Cycles to 75% energy', 'Cycles to 70% energy', 'Cycles to 60% energy', 'Cycles to 50% energy', 'Run ID',
        'Electrolyte to press (s)', 'Electrolyte to electrode (s)', 'Electrode to protection (s)', 'Press to protection (s)']

cycling_data = r"G:\Limit\Lina Scholz\Cell Data\batch.lisc_gen14.json"
alignment_data = r"C:\lisc_gen14\data\alignment.csv"

# Open and load the JSON file
with open(cycling_data, 'r') as file:
    json_data = json.load(file)
    # Extract the data associated with the key "data"
    if "data" in json_data:
        cell_data = json_data["data"]

data = pd.read_csv(alignment_data)
performance_numbers = ['First formation efficiency (%)', 'First formation specific discharge capacity (mAh/g)',
                       'Initial specific discharge capacity (mAh/g)', 'Initial efficiency (%)', 'Capacity loss (%)',
                       'Last specific discharge capacity (mAh/g)', 'Last efficiency (%)',
                       'Formation average voltage (V)', 'Formation average current (A)', 'Initial delta V (V)',
                       'Cycles to 90% capacity', 'Cycles to 80% capacity', 'Cycles to 70% capacity']
# 'Specific discharge capacity (mAh/g)'
spec_disc_capacity = 'Specific discharge capacity (mAh/g)'
data['Specific discharge capacity 150th (mAh/g)'] = None # initialize column
data['Specific discharge capacity 5th (mAh/g)'] = None # initialize column
data["Fade rate 5-20 cycles (%/cycle)"] = None # initialize column
data["Fade rate 5-50 cycles (%/cycle)"] = None # initialize column
# data['Electrodes center'] = None # initialize column
for string in performance_numbers:
    data[string] = None # initialize columns

for cell in cell_data:
    number = int(cell["Sample ID"].split("_")[-1])
    for i in range(len(performance_numbers)):
        data.loc[data['cell'] == number, performance_numbers[i]] = cell[performance_numbers[i]]
    data.loc[data['cell'] == number,
             'Specific discharge capacity 150th (mAh/g)'] = cell['Specific discharge capacity (mAh/g)'][150]
    data.loc[data['cell'] == number,
             'Specific discharge capacity 5th (mAh/g)'] = cell['Specific discharge capacity (mAh/g)'][5]
    # Rate rate between cycles 5-20 and 5-50
    data.loc[data['cell'] == number,
             "Fade rate 5-20 cycles (%/cycle)"] = np.diff(cell["Normalised discharge capacity (%)"])[5:20].mean()
    data.loc[data['cell'] == number,
             "Fade rate 5-50 cycles (%/cycle)"] = np.diff(cell["Normalised discharge capacity (%)"])[5:50].mean()
data = data.dropna()

# calculate distances between main components: spring, anode, cathode, spacer
d27_list = []
d28_list = []
d67_list = []
d68_list = []
d78_list = []
electrodes_x = []
electrodes_y = []
electrodes_z = []
electrodes_spring = []
electrodes_spacer = []
for cell in data["cell"].unique():
    cell_df = data[data['cell'] == cell]
    d27 = round(math.sqrt((cell_df["x2"].values - cell_df["x6"].values)**2
                        + (cell_df["y2"].values - cell_df["y6"].values)**2), 3)
    d28 = round(math.sqrt((cell_df["x2"].values - cell_df["x8"].values)**2
                        + (cell_df["y2"].values - cell_df["y8"].values)**2), 3)
    d67 = round(math.sqrt((cell_df["x6"].values - cell_df["x7"].values)**2
                        + (cell_df["y6"].values - cell_df["y7"].values)**2), 3)
    d68 = round(math.sqrt((cell_df["x6"].values - cell_df["x8"].values)**2
                        + (cell_df["y6"].values - cell_df["y8"].values)**2), 3)
    d78 = round(math.sqrt((cell_df["x7"].values - cell_df["x8"].values)**2
                        + (cell_df["y7"].values - cell_df["y8"].values)**2), 3)
    e_x = round((cell_df["x2"].values[0] + cell_df["x6"].values[0])/2, 3)
    e_y = round((cell_df["y2"].values[0] + cell_df["y6"].values[0])/2, 3)
    e_z = round(math.sqrt(e_x**2 + e_y**2), 3)
    e_spring = round(math.sqrt((e_x - cell_df["x8"].values)**2
                               + (e_y - cell_df["y8"].values)**2), 3)
    e_spacer = round(math.sqrt((e_x - cell_df["x7"].values)**2
                               + (e_y - cell_df["y7"].values)**2), 3)
    d27_list.append(d27)
    d28_list.append(d28)
    d67_list.append(d67)
    d68_list.append(d68)
    d78_list.append(d78)
    electrodes_x.append(e_x)
    electrodes_y.append(e_y)
    electrodes_z.append(e_z)
    electrodes_spring.append(e_spring)
    electrodes_spacer.append(e_spacer)

data["d26"] = data["z_electrodes"]
data["d27"] = d27_list
data["d28"] = d28_list
data["d67"] = d67_list
data["d68"] = d68_list
data["d78"] = d78_list
data["electrodes_x"] = electrodes_x
data["electrodes_y"] = electrodes_y
data["electrodes_to_press"] = electrodes_z
data["electrodes_spring"] = electrodes_spring
data["electrodes_spacer"] = electrodes_spacer

# general score
def normalize(values, c, k): # c: middle point / turning point; k: slope
    # Parameter for sigmoid function
    return 1 / (1 + np.exp(-k * (values - c)))

# c = 98 for intersection area
# c = 1.5 for electrode alignment
# c = 1 # for electrode to spring alignment
data["electrodes_normalized"] = normalize(data["d26"].to_numpy(), 1.5, -1)
data["electrodes_to_press_normalized"] = normalize(data["electrodes_to_press"].to_numpy(), 1.5, -1)
data["electrode_to_spring_normalized"] = normalize(data["electrodes_spring"].to_numpy(), 1.5, -1)
data["area_normalized"] = normalize(data["intersection_area"].to_numpy(), 97, 1)

data["alignment_score_1"] = data["electrodes_normalized"] * data["electrode_to_spring_normalized"] * 100
data["alignment_score_2"] = data["electrodes_normalized"] * data["electrodes_to_press_normalized"] * 100

# Save the plot as a JPG file named by the cell number
data_dir = os.path.join("C:/lisc_gen14/data", "plot")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
with pd.ExcelWriter(os.path.join(data_dir, "performance.xlsx")) as writer:
    data.to_excel(writer, sheet_name='performance', index=False)
data.to_csv(os.path.join(data_dir, "performance.csv"), index=False)

#%% Find any CORRELATION

data_analysis = data[["Fade rate 5-50 cycles (%/cycle)", "Cycles to 70% capacity",
                      "Initial specific discharge capacity (mAh/g)", "Specific discharge capacity 150th (mAh/g)",
                      "z2", "z6", "z8", "intersection_area", "d26", "electrodes_to_press",
                      "electrodes_spring"]]

# correlation matrix
correlation_matrix = data_analysis.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.tight_layout()
plt.show()

# Define the output path
data_dir = os.path.join("C:/lisc_gen14", "data")  # Replace "your_path" with your desired path
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Scatter Plot
# Degradation (Pressure)
fig = make_subplots(rows=3, cols=3)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data["d28"],
        y=data["Fade rate 5-50 cycles (%/cycle)"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d28: %{x}<br>" + "Fade rate 5-50 cycles (%/cycle): %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data["d68"],
        y=data["Fade rate 5-50 cycles (%/cycle)"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d68: %{x}<br>" + "Fade rate 5-50 cycles (%/cycle): %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=data["electrodes_spring"],
        y=data["Fade rate 5-50 cycles (%/cycle)"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("electrodes_spring: %{x}<br>" + "Fade rate 5-50 cycles (%/cycle): %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=data["d28"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d28: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=data["d68"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d68: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=2, col=2)
fig.add_trace(
    go.Scatter(
        x=data["electrodes_spring"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("electrodes_spring: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=2, col=3)
fig.add_trace(
    go.Scatter(
        x=data["d27"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d27: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=3, col=1)
fig.add_trace(
    go.Scatter(
        x=data["d67"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("d67: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=3, col=2)
fig.add_trace(
    go.Scatter(
        x=data["electrodes_spacer"],
        y=data["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data["d26"], colorscale='viridis_r', colorbar=dict(title="electrodes [mm]")),
        text=data["cell"],  # Add "cell" values for hover
        hovertemplate=("electrodes_spacer: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=3, col=3)

# Update axis
fig.update_xaxes(title_text="anode to spring [mm]", row=1, col=1)
fig.update_yaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=1)
fig.update_xaxes(title_text="cathode to spring [mm]", row=1, col=2)
fig.update_yaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=2)
fig.update_xaxes(title_text="electrodes to spring [mm]", row=1, col=3)
fig.update_yaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=3)
fig.update_xaxes(title_text="anode to spring [mm]", row=2, col=1)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=2, col=1)
fig.update_xaxes(title_text="cathode to spring [mm]", row=2, col=2)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=2, col=2)
fig.update_xaxes(title_text="electrodes to spring [mm]", row=2, col=3)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=2, col=3)
fig.update_xaxes(title_text="anode to spacer [mm]", row=3, col=1)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=3, col=1)
fig.update_xaxes(title_text="cathode to spacer [mm]", row=3, col=2)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=3, col=2)
fig.update_xaxes(title_text="electrodes to spacer [mm]", row=3, col=3)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=3, col=3)

# Update layout
fig.update_layout(
    title="Scatter Plots of Fade Rate and Cycles to 70% capacity vs Spring Alignment",
    height=900,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Save the figure as a .jpg
output_file = os.path.join(data_dir, "Degradation(pressure)_correlation.jpg")
# fig.write_image(output_file, format="jpg")

# Capacity (Electrodes)
fig = make_subplots(rows=2, cols=3)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data["Initial specific discharge capacity (mAh/g)"],
        y=data["intersection_area"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Ini. spec. disc. capacity: %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data["Specific discharge capacity 150th (mAh/g)"],
        y=data["intersection_area"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Spec. disc. capacity 150th (mAh/g): %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=data["Fade rate 5-50 cycles (%/cycle)"],
        y=data["intersection_area"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Fade rate 5-50 cycles (%/cycle): %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=data["Initial specific discharge capacity (mAh/g)"],
        y=data["d26"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Ini. spec. disc. capacity: %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=data["Specific discharge capacity 150th (mAh/g)"],
        y=data["d26"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Spec. disc. capacity 150th (mAh/g): %{x}<br>" +
                        "d26: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=2)
fig.add_trace(
    go.Scatter(
        x=data["Fade rate 5-50 cycles (%/cycle)"],
        y=data["d26"],
        mode='markers',
        marker=dict(color=data["electrodes_to_press"]*100, colorscale='viridis_r', colorbar=dict(title="electrodes to center [mm]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Fade rate 5-50 cycles (%/cycle): %{x}<br>" +
                        "d26: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=3)
# Update axis
fig.update_xaxes( title_text="Initial specific discharge capacity (mAh/g)", row=1, col=1)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=1)
fig.update_xaxes(title_text="Specific discharge capacity 150th (mAh/g)", row=1, col=2)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=2)
fig.update_xaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=3)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=3)
fig.update_xaxes( title_text="Initial specific discharge capacity (mAh/g)", row=2, col=1)
fig.update_yaxes(title_text="Electrode alignment [mm]", row=2, col=1)
fig.update_xaxes(title_text="Specific discharge capacity 150th (mAh/g)", row=2, col=2)
fig.update_yaxes(title_text="Electrode alignment [mm]", row=2, col=2)
fig.update_xaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=2, col=3)
fig.update_yaxes(title_text="Electrode alignment [mm]", row=2, col=3)

# Update layout
fig.update_layout(
    title="Scatter Plots of Specific Discharge Capacities and Fade rate vs Intersection Area",
    height=800,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Save the figure as a .jpg
output_file = os.path.join(data_dir, "Capacity(electrodes)_correlation.jpg")
# fig.write_image(output_file, format="jpg")

# Alignment Score
fig = make_subplots(rows=2, cols=3)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data["Specific discharge capacity 150th (mAh/g)"],
        y=data["alignment_score_1"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [%]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Spec. dis. capacity 150th (mAh/g): %{x}<br>" +
                        "Alignment Score 1 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data["Cycles to 70% capacity"],
        y=data["alignment_score_1"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [R]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Cycles to 70% capacity: %{x}<br>" +
                        "Alignment Score 1 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=data["Fade rate 5-50 cycles (%/cycle)"],
        y=data["alignment_score_1"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [R]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Fade rate 5-50 cycles (%/cycle): %{x}<br>" +
                        "Alignment Score 1 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=data["Specific discharge capacity 150th (mAh/g)"],
        y=data["alignment_score_2"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [%]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Spec. dis. capacity 150th (mAh/g): %{x}<br>" +
                        "Alignment Score 2 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=data["Cycles to 70% capacity"],
        y=data["alignment_score_2"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [R]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Cycles to 70% capacity: %{x}<br>" +
                        "Alignment Score 2 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=2)
fig.add_trace(
    go.Scatter(
        x=data["Fade rate 5-50 cycles (%/cycle)"],
        y=data["alignment_score_2"],
        mode='markers',
        marker=dict(color=data["intersection_area"], colorscale='viridis_r', colorbar=dict(title="intersection_area [R]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Fade rate 5-50 cycles (%/cycle): %{x}<br>" +
                        "Alignment Score 2 [%]: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=2, col=3)

fig.update_xaxes(title_text="Specific discharge capacity 150th (mAh/g)", row=1, col=1)
fig.update_yaxes(title_text="Alignment Score 1 [%]", row=1, col=1)
fig.update_xaxes(title_text="Cycles to 70% capacity", row=1, col=2)
fig.update_yaxes(title_text="Alignment Score 1 [%]", row=1, col=2)
fig.update_xaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=3)
fig.update_yaxes(title_text="Alignment Score 1 [%]", row=1, col=3)
fig.update_xaxes(title_text="Specific discharge capacity 150th (mAh/g)", row=2, col=1)
fig.update_yaxes(title_text="Alignment Score 2 [%]", row=2, col=1)
fig.update_xaxes(title_text="Cycles to 70% capacity", row=2, col=2)
fig.update_yaxes(title_text="Alignment Score 2 [%]", row=2, col=2)
fig.update_xaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=2, col=3)
fig.update_yaxes(title_text="Alignment Score 2 [%]", row=2, col=3)

# Update layout
fig.update_layout(
    title="Scatter Plots of Alignment Score",
    height=800,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Capacity (Electrodes)
fig = make_subplots(rows=1, cols=2)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data["Cycles to 70% capacity"],
        y=data["z8"],
        mode='markers',
        marker=dict(color=data["z8"], colorscale='viridis_r', colorbar=dict(title="intersection_area [%]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Cycles to 70% capacity: %{x}<br>" +
                        "z8: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data["Fade rate 5-50 cycles (%/cycle)"],
        y=data["z8"],
        mode='markers',
        marker=dict(color=data["z8"], colorscale='viridis_r', colorbar=dict(title="intersection_area [%]")),
        text=("Cell: " + data["cell"].astype(str) + "<br>" +
              "d28: " + data["d28"].astype(str) + "<br>" +
              "d27: " + data["d27"].astype(str) + "<br>" +
              "d67: " + data["d67"].astype(str)),
        hovertemplate=("Spec. dis. capacity 150th (mAh/g): %{x}<br>" +
                        "z8: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)

fig.update_xaxes(title_text="Cycles to 70% capacity", row=1, col=1)
fig.update_yaxes(title_text="Spring [mm]", row=1, col=1)
fig.update_xaxes(title_text="Fade rate 5-50 cycles (%/cycle)", row=1, col=2)
fig.update_yaxes(title_text="Spring [mm]", row=1, col=2)

# Update layout
fig.update_layout(
    title="Scatter Plots of Alignment Score",
    height=500,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Save the figure as a .jpg
output_file = os.path.join(data_dir, "Alignment_number_correlation.jpg")
# fig.write_image(output_file, format="jpg")

#%% PCA

pca = False

if pca:
    df_normalized = (data_analysis - data_analysis.mean()) / data_analysis.std()
    pca = PCA(n_components=data_analysis.shape[1])
    pca.fit(df_normalized)
    pca_reduced = pca.fit_transform(df_normalized)
    # Reformat and view results
    loadings = pd.DataFrame(pca.components_.T,
                            columns=['PC%s' % _ for _ in range(len(df_normalized.columns))], index=data_analysis.columns)
    # Append the principle components for each entry to the dataframe
    pca_data = data_analysis.copy()
    for i in range(0, data_analysis.shape[1]):
        pca_data['PC' + str(i + 1)] = pca_reduced[:, i]

    # Do a scree plot
    ind = np.arange(0, data_analysis.shape[1])
    (fig, ax) = plt.subplots(figsize=(8, 6))
    sns.pointplot(x=ind, y=pca.explained_variance_ratio_)
    ax.set_title('Scree plot')
    ax.set_xticks(ind)
    ax.set_xticklabels(ind)
    ax.set_xlabel('Component Number')
    ax.set_ylabel('Explained Variance')
    # plt.show()
    # Plot a variable factor map for the first two dimensions.
    (fig, ax) = plt.subplots(figsize=(8, 8))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,
                0,  # Start the arrow at the origin
                pca.components_[0, i],  # 0 for PC1
                pca.components_[1, i],  # 1 for PC2
                head_width=0.1,
                head_length=0.1)
        plt.text(pca.components_[0, i] + 0.05,
                pca.components_[1, i] + 0.05,
                data_analysis.columns.values[i])
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    plt.axis('equal')
    ax.set_title('Correlation Circle')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.axhline(0, linestyle=":", color="grey")
    ax.axvline(0, linestyle=":", color="grey")
    # plt.show()







# %%
