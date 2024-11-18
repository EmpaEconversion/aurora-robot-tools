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

cycling_data = r"G:\Limit\Lina Scholz\Data\batch.lisc_gen14.json"
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
for string in performance_numbers:
    data[string] = None

for cell in cell_data:
    number = int(cell["Sample ID"].split("_")[-1])
    for i in range(len(performance_numbers)):
        data.loc[data['cell'] == number, performance_numbers[i]] = cell[performance_numbers[i]]
data = data.dropna()

# calculate distances between main components
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

# scatter plot
# degradration (spring)
fig = px.scatter_matrix(
    data_analysis,
    dimensions=['d28', "d68", 'Cycles to 70% capacity', "Formation average current (A)", 'Formation average voltage (V)'],
    title="Pressure Distribution: spring vs. electrodes",
    hover_data=["cell", "d26", "intersection_area", 'First formation efficiency (%)',
                'First formation specific discharge capacity (mAh/g)',
                'Initial specific discharge capacity (mAh/g)', 'Initial efficiency (%)',
                'Last specific discharge capacity (mAh/g)', 'Last efficiency (%)', 'Formation average voltage (V)',
                'Formation average current (A)', 'Initial delta V (V)'],
    color="d26",
    labels={
        "d28": "D28 [mm]",
        "d68": "D68 [mm]",
        'Formation average current (A)': "Form. av. curr. (A)",
        "Cycles to 70% capacity": "Cyc. to 70%",
        'Formation average voltage (V)': "Form. av. volt. (V)"},
    color_continuous_scale="Viridis"
)
fig.update_layout(coloraxis_colorbar=dict(title="anode vs. cathode"))
fig.update_traces(diagonal_visible=True) # Zeigt Histogramme auf der Diagonale
fig.show()
# capacity (electrodes)
fig = px.scatter_matrix(
    data_analysis,
    dimensions=['intersection_area', 'Last specific discharge capacity (mAh/g)',
                'Initial specific discharge capacity (mAh/g)'],
    title="Electrode Alignment",
    hover_data=["cell", "d26", "d28", "d68", "intersection_area", 'First formation efficiency (%)',
                'First formation specific discharge capacity (mAh/g)',
                'Initial efficiency (%)', 'Last efficiency (%)', 'Formation average voltage (V)',
                'Formation average current (A)', 'Initial delta V (V)', 'Cycles to 70% capacity', 'Capacity loss (%)'],
    color="d68",
    labels={
        "intersection_area": "Intersec. Area",
        "Last specific discharge capacity (mAh/g)": "Last spec. disc. cap. (mAh/g)",
        "Initial specific discharge capacity (mAh/g)": "Init. Spec. Cap. (mAh/g)"},
    color_continuous_scale="Viridis"
)
fig.update_layout(coloraxis_colorbar=dict(title="cathode vs. spring"))
fig.update_traces(diagonal_visible=True)  # Zeigt Histogramme auf der Diagonale
fig.show()

# PCA
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
plt.show()
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
plt.show()

# TODO: The second 1 over C cycle (is used for normation and is the 5th cycle overall) -> correlate to anode/cathode

# cluster analysis
Z = linkage(data_analysis[['d26', 'd27', 'd28', 'd67', "d68", "d78"]], method='ward')
dendrogram(Z)

# multivariate Analysis
X = data_analysis[['d26', 'd28', "d68", "d78"]]
y = data_analysis['Cycles to 70% capacity'].values
X = sm.add_constant(X)
# model = sm.OLS(y, X).fit()
# print(model.summary())

print("End")
