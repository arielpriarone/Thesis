from calendar import c
from turtle import color
from matplotlib.colors import Colormap
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.fft import rfft, rfftfreq

# Read the CSV files into dataframes
df_features = pd.read_csv(r"C:\Users\ariel\Documents\Courses\Tesi\Code\train_data.csv", sep='\t').dropna(axis=1)
df_timeseries = pd.read_csv(r"C:\Users\ariel\Documents\Courses\Tesi\Code\timeseries_data.csv", sep='\t').dropna(axis=1)

# standardise the features
df_features.iloc[:, 1:] = (df_features.iloc[:, 1:] - df_features.iloc[:, 1:].mean()) / df_features.iloc[:, 1:].std()

# Create a plot for dataframe of features
fig = plt.figure()
ax1 = fig.add_subplot(111)
colormap = cm.get_cmap('cool')
i=0
for timestamp in df_features['Timestamp'].unique():
    df_features_timestamp = df_features[df_features['Timestamp'] == timestamp]
    if i < 20:
        color = 'blue'
    elif i < 40:
        color = 'red'
    elif i < 60:
         color = 'green'
    else:
        color = 'orange'  
    ax1.plot(df_features_timestamp.columns[1:], df_features_timestamp.values[0, 1:], label=timestamp, color = color)
    i+=1
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Features')
ax1.tick_params(labelrotation=45)
ax1.legend()

#Create a plot for dataframe of timeseries
fig = plt.figure()
ax2 = fig.add_subplot(111)
i=0
for timestamp in df_timeseries['Timestamp'].unique():
    if i < 20:
        color = 'blue'
    elif i < 40:
        color = 'red'
    elif i < 60:
         color = 'green'
    else:
        color = 'orange'  
    df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
    ax2.plot(df_timeseries_timestamp.columns[1:], df_timeseries_timestamp.values[0, 1:], label=timestamp, color = color)
    i+=1
fig.tight_layout()

# plot the spectrum of the timeseries
fig, axs = plt.subplots()
sampling_rate = 5000 # Hz
N=6000 # number of samples
i=0
for timestamp in df_timeseries['Timestamp'].unique():
    if i < 20:
        color = 'blue'
    elif i < 40:
        color = 'red'
    elif i < 60:
         color = 'green'
    else:
        color = 'orange'   
    df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
    # normalize the timeseries
    fourier_timeserie = (df_timeseries_timestamp.values[0, 1:]-np.mean(df_timeseries_timestamp.values[0, 1:]))
    # preprocess hann window
    hann_window = np.hanning(N)
    fourier_timeserie = np.multiply(fourier_timeserie, hann_window)
    # plot the spectrum
    axs.plot(rfftfreq(N, d=1/sampling_rate), 2*np.abs(rfft(fourier_timeserie))/N,color=color)
    i+=1
axs.set_xlabel('Frequency[Hz]')
axs.set_ylabel('Amplitude')
axs.set_yscale('log')

# Show the plots
plt.show()

