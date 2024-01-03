from turtle import color
from matplotlib.colors import Colormap
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

# Read the CSV files into dataframes
df_features = pd.read_csv(r"C:\Users\ariel\Documents\Courses\Tesi\Code\train_data.csv", sep='\t').dropna(axis=1)
df_timeseries = pd.read_csv(r"C:\Users\ariel\Documents\Courses\Tesi\Code\timeseries_data.csv", sep='\t').dropna(axis=1)

# standardise the features
df_features.iloc[:, 1:] = (df_features.iloc[:, 1:] - df_features.iloc[:, 1:].mean()) / df_features.iloc[:, 1:].std()

# Create a 3D plot for dataframe of features
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
colormap = cm.get_cmap('cool')
i=0
for timestamp in df_features['Timestamp'].unique():
    df_features_timestamp = df_features[df_features['Timestamp'] == timestamp]
    if i < 10:
        color = 'blue'
    else:
        color = 'red'
    ax1.plot(df_features_timestamp.columns[1:], df_features_timestamp.values[0, 1:], label=timestamp, color = color)
    i+=1
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Features')
ax1.tick_params(labelrotation=45)
ax1.legend()

import src
fig, ax = plt.subplots()
for timestamp in df_timeseries['Timestamp'].unique():
    df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
    result, freqs, _ = src.features.FFT(df_timeseries_timestamp.to_xarray(), 5000, preproc=None)
    ax.plot(freqs, result)

# Create a 3D plot for dataframe of timeseries
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# for timestamp in df_timeseries['Timestamp'].unique():
#     df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
#     ax2.plot(df_timeseries_timestamp.columns[1:], df_timeseries_timestamp.values[0, 1:], label=timestamp)

# Show the plots
plt.show()

