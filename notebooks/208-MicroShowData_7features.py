from cProfile import label
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
from rich import print

src.visualization.set_matplotlib_params()


# Read the CSV files into dataframes
df_novelty = np.reshape(pd.read_csv(r"data\processed\ETEL_Test3\test_data_20240227.csv", sep='\t')['Novelty'].values*100, (9,10))

print(df_novelty)

# Plot the novelty
labels = ["acc: 6, spd: 0.4",
          "acc: 6, spd: 0.6",
          "acc: 6, spd: 0.8",
          "acc: 3, spd: 0.8",
          "acc: 4, spd: 0.8",
          "acc: 5, spd: 0.8",
          "acc: 4, spd: 0.6",
          "acc: 3, spd: 0.4",
          "acc: 6, spd: 0.2"]
fig, ax = plt.subplots()
for i in range(df_novelty.shape[0]):
    ax.plot(range(i*10+1,i*10+11),df_novelty[i],label = labels[i])
# Set the labels
ax.set_xlabel('Snapshot')
ax.set_ylabel('Novelty metric [%]')
# Set the legend
ax.legend()
# auto tick format
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
# set both grids
ax.grid(True, which='both')



plt.show()