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
import statsmodels.api as sm
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

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
    # data.loc[data['cell'] == number,
             # 'Electrodes center'] = cell[''] # TODO                                  
data = data.dropna()

# calculate distances between main components: spring, anode, cathode, spacer
d27_list = []
d28_list = []
d67_list = []
d68_list = []
d78_list = []
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
    d27_list.append(d27)
    d28_list.append(d28)
    d67_list.append(d67)
    d68_list.append(d68)
    d78_list.append(d78)

data["d26"] = data["z_electrodes"]
data["d27"] = d27_list
data["d28"] = d28_list
data["d67"] = d67_list
data["d68"] = d68_list
data["d78"] = d78_list

# Save the plot as a JPG file named by the cell number
data_dir = os.path.join("C:/lisc_gen14/data", "plot")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
with pd.ExcelWriter(os.path.join(data_dir, "performance.xlsx")) as writer:
    data.to_excel(writer, sheet_name='performance', index=False)
data.to_csv(os.path.join(data_dir, "performance.csv"), index=False)

#%% Performance



#%% Find any CORRELATION

data_analysis = data.copy()
data_analysis = data_analysis.drop(columns=["press", "z_electrodes",
                                   "x2", "y2", "z2", "x6", "y6", "z6", "x7", "y7", "z7", "x8", "y8", "z8",
                                   'Cycles to 90% capacity', 'Cycles to 80% capacity',
                                   "x1", "y1", "z1", "x4", "y4", "z4", "x9", "y9", "z9"])

# correlation matrix
correlation_matrix = data_analysis.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Scatter Plot
# Degradation (Pressure)
fig = make_subplots(rows=2, cols=2)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data_analysis["d28"],
        y=data_analysis["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data_analysis["d26"], colorscale='Viridis', colorbar=dict(title="electrodes [mm]")),
        text=data_analysis["cell"],  # Add "cell" values for hover
        hovertemplate=("d28: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data_analysis["d68"],
        y=data_analysis["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data_analysis["d26"], colorscale='Viridis', colorbar=dict(title="electrodes [mm]")),
        text=data_analysis["cell"],  # Add "cell" values for hover
        hovertemplate=("d68: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=data_analysis["d27"],
        y=data_analysis["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data_analysis["d26"], colorscale='Viridis', colorbar=dict(title="electrodes [mm]")),
        text=data_analysis["cell"],  # Add "cell" values for hover
        hovertemplate=("d27: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=2, col=1)
fig.add_trace(
    go.Scatter(
        x=data_analysis["d67"],
        y=data_analysis["Cycles to 70% capacity"],
        mode='markers',
        marker=dict(color=data_analysis["d26"], colorscale='Viridis', colorbar=dict(title="electrodes [mm]")),
        text=data_analysis["cell"],  # Add "cell" values for hover
        hovertemplate=("d67: %{x}<br>" + "Cycles to 70%: %{y}<br>" +
                       "d26: %{marker.color}<br>" + "Cell: %{text}<extra></extra>"),
        showlegend=False),
    row=2, col=2)
# Update axis
fig.update_xaxes(title_text="anode to spring [mm]", row=1, col=1)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=1, col=1)

fig.update_xaxes(title_text="cathode to spring [mm]", row=1, col=2)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=1, col=2)

fig.update_xaxes(title_text="anode to spacer [mm]", row=2, col=1)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=2, col=1)

fig.update_xaxes(title_text="cathode to spacer [mm]", row=2, col=2)
fig.update_yaxes(title_text="Cycles to 70% capacity", row=2, col=2)

# Update layout
fig.update_layout(
    title="Scatter Plots of Specific Discharge Capacities vs Intersection Area",
    height=500,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()

# Capacity (Electrodes)
fig = make_subplots(rows=1, cols=3)
# Add scatter plots
fig.add_trace(
    go.Scatter(
        x=data_analysis["Specific discharge capacity 5th (mAh/g)"],
        y=data_analysis["intersection_area"],
        mode='markers',
        marker=dict(color=data_analysis["d68"], colorscale='Viridis', colorbar=dict(title="d68 [mm]")),
        text=("Cell: " + data_analysis["cell"].astype(str) + "<br>" +
              "d28: " + data_analysis["d28"].astype(str) + "<br>" +
              "d27: " + data_analysis["d27"].astype(str) + "<br>" +
              "d67: " + data_analysis["d67"].astype(str)),
        hovertemplate=("Spec. disc. capacity 5th (mAh/g): %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=data_analysis["Specific discharge capacity 150th (mAh/g)"],
        y=data_analysis["intersection_area"],
        mode='markers',
        marker=dict(color=data_analysis["d68"], colorscale='Viridis', colorbar=dict(title="d68 [mm]")),
        text=("Cell: " + data_analysis["cell"].astype(str) + "<br>" +
              "d28: " + data_analysis["d28"].astype(str) + "<br>" +
              "d27: " + data_analysis["d27"].astype(str) + "<br>" +
              "d67: " + data_analysis["d67"].astype(str)),
        hovertemplate=("Spec. disc. capacity 150th: %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=data_analysis["Initial specific discharge capacity (mAh/g)"],
        y=data_analysis["intersection_area"],
        mode='markers',
        marker=dict(color=data_analysis["d68"], colorscale='Viridis', colorbar=dict(title="d68 [mm]")),
        text=("Cell: " + data_analysis["cell"].astype(str) + "<br>" +
              "d28: " + data_analysis["d28"].astype(str) + "<br>" +
              "d27: " + data_analysis["d27"].astype(str) + "<br>" +
              "d67: " + data_analysis["d67"].astype(str)),
        hovertemplate=("Ini. spec. disc. capacity: %{x}<br>" +
                        "Intersection Area: %{y}<br>" +
                        "%{text}<extra></extra>"),
        showlegend=False),
    row=1, col=3)
# Update axis
fig.update_xaxes(title_text="Specific discharge capacity 5th (mAh/g)", row=1, col=1)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=1)

fig.update_xaxes(title_text="Specific discharge capacity 150th (mAh/g)", row=1, col=2)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=2)

fig.update_xaxes( title_text="Initial specific discharge capacity (mAh/g)", row=1, col=3)
fig.update_yaxes(title_text="Intersection area [%]", row=1, col=3)
# Update layout
fig.update_layout(
    title="Scatter Plots of Specific Discharge Capacities vs Intersection Area",
    height=300,  # Niedriger machen
    margin=dict(l=50, r=50, t=50, b=50))
fig.show()


#%% PCA

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

# TODO: The second 1 over C cycle (is used for normation and is the 5th cycle overall) -> correlate to anode/cathode




