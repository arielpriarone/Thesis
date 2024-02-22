from calendar import c
from turtle import color
from matplotlib.colors import Colormap
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from scipy.fft import rfft, rfftfreq
import matplotlib.ticker as ticker
import src
import matplotlib as mpl

src.visualization.set_matplotlib_params()
mpl.rcParams['lines.linewidth'] = 0.5

# Read the CSV files into dataframes
df_features = pd.read_csv(r"data\processed\ETEL_Test1\train_data.csv", sep='\t').dropna(axis=1)
df_timeseries = pd.read_csv(r"data\processed\ETEL_Test1\timeseries_data.csv", sep='\t').dropna(axis=1)

# standardise the features
df_features.iloc[:, 1:] = (df_features.iloc[:, 1:] - df_features.iloc[:, 1:].mean()) / df_features.iloc[:, 1:].std()

def setcolor(i):
    test_intervals = [99, 199, 299]
    if i < test_intervals[0]:
        return 'blue'
    elif i < test_intervals[1]:
        return 'red'
    elif i < test_intervals[2]:
        return 'green'
    else:
        return 'orange' 
class labelsetter(): 
    def __init__(self):
        self.j = 1
        self.memcolor = None
    def getlabel(self,i):
        color = setcolor(i)
        if self.memcolor != color:
            label = "profile " + str(self.j)
            self.j+=1
        else:
            label = "_nolegend_"
        self.memcolor = color
        return label
# Create a plot for dataframe of features
fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.subplots_adjust(top=0.954,
bottom=0.278,
left=0.057,
right=0.995,
hspace=0.2,
wspace=0.2)
colormap = cm.get_cmap('cool')
i=0
labelSetter = labelsetter()
for timestamp in df_features['Timestamp'].unique():
    df_features_timestamp = df_features[df_features['Timestamp'] == timestamp]
    color = setcolor(i) 
    label = labelSetter.getlabel(i)
    ax1.plot(df_features_timestamp.columns[1:], df_features_timestamp.values[0, 1:], label=label, color = color)
    i+=1
ax1.set_ylabel('Value')
ax1.set_xlim(0, 66)
ax1.tick_params(labelrotation=90)
ax1.legend()

# plot a scatter plot of the features
src.visualization.set_matplotlib_params()
fig, ax = plt.subplots(2,3)
features_plot_set = [5,6,25,26]

ax_row = 0
ax_col = 0
for j in features_plot_set:
    for k in features_plot_set[features_plot_set.index(j) +1:]:
        i=0
        for timestamp in df_features['Timestamp'].unique():
            df_features_timestamp = df_features[df_features['Timestamp'] == timestamp]
            color = setcolor(i) 
            ax[ax_row, ax_col].scatter(df_features_timestamp.values[0, j], df_features_timestamp.values[0, k], marker='.' ,color=color)
            ax[ax_row, ax_col].set_xlabel(df_features.columns[j])
            ax[ax_row, ax_col].set_ylabel(df_features.columns[k])
            ax[ax_row, ax_col].set_xticks([])
            ax[ax_row, ax_col].set_yticks([])
            i+=1
        ax_col+=1
        print(ax_col)
        if ax_col == 3:
            ax_col = 0
            ax_row += 1
    
plt.tight_layout()
plt.show()
exit()

#Create a plot for dataframe of timeseries
fig = plt.figure()
ax2 = fig.add_subplot(111)
i=0
for timestamp in df_timeseries['Timestamp'].unique():
    color = setcolor(i)    
    df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
    ax2.plot(df_timeseries_timestamp.columns[1:], df_timeseries_timestamp.values[0, 1:], label=timestamp, color = color)
    i+=1
ax2.xaxis.set_major_locator(ticker.AutoLocator())
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.set_xlabel('time')
ax2.set_ylabel('voltage')
fig.tight_layout()

#Create a plot for dataframe of timeseries
fig = plt.figure()
ax2 = fig.add_subplot(111)
i=0
labelSetter = labelsetter()
for timestamp in df_timeseries['Timestamp'].unique():
    color = setcolor(i) 
    df_timeseries_timestamp = df_timeseries[df_timeseries['Timestamp'] == timestamp]
    label = labelSetter.getlabel(i)
    ax2.plot(df_timeseries_timestamp.columns[1:], np.divide(df_timeseries_timestamp.values[0, 1:]-1.235,0.24), label=label, color = color)
    i+=1
ax2.xaxis.set_major_locator(ticker.AutoLocator())
ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.set_xlabel('time [ms]')
ax2.set_ylabel('acceleration [g]')
ax2.legend(loc='upper right')
fig.tight_layout()

# plot the spectrum of the timeseries
fig, axs = plt.subplots()
sampling_rate = 5000 # Hz
N=6000 # number of samples
i=0
for timestamp in df_timeseries['Timestamp'].unique():
    color = setcolor(i) 
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

mean_value = df_timeseries.iloc[:, 1:].mean().mean()
print(mean_value)

# Show the plots
plt.show()

