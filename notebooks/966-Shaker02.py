from os import times
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from rich import print
import src
src.visualization.set_matplotlib_params()

filepath = r"data\processed\Shaker_test02\test_data_shaker.csv"
tests_separators = [10,10,10,10,10,10,10,10,10,10,10]; # number of samples per test
tests_names = [ "$V_{pp} =  580$mV",
                "$V_{pp} = 1000$mV",
                "$V_{pp} = 1980$mV",
                "$V_{pp} = 1540$mV",
                "$V_{pp} = 2000$mV",
                "$V_{pp} =    0$mV",
                "$V_{pp} =  800$mV",
                "$V_{pp} =  200$mV",
                "$V_{pp} = 2120$mV",
                "$V_{pp} = 2000$mV",
                "$V_{pp} = 1540$mV (second wave)"]

# %% Load the data
data = pd.read_csv(filepath, sep='\t')
print(data.keys())  
print(data.head())

NoveltyMetric = data["Novelty"].to_numpy()
Timestamp_all = data["Timestamp"].to_numpy()
# %% Split the data
tests = []
timestamps = []
start = 0
for i, samples in enumerate(tests_separators):
    end = start + samples
    print(f"slicing from {start} to {end}")
    tests.append(NoveltyMetric[start:end])
    timestamps.append((Timestamp_all[start:end]))
    start = end

# %% plot the data
fig, ax = plt.subplots()
plt.subplots_adjust(top=0.731,
bottom=0.08,
left=0.135,
right=0.974,
hspace=0.2,
wspace=0.2)

for i, test in enumerate(tests):
    xlabels = [dt.datetime.fromisoformat(ts) for ts in timestamps[i]]
    ax.plot(timestamps[i],test*100, label=tests_names[i])
    print(f"Test {i} - {tests_names[i]}")
    print(f"Novelty: {(test)}")
ax.axhline(y=5, color='k', linestyle='-.', label='threshold')
ax.set_xticklabels([])
ax.set_xlabel("snapshots")
ax.set_ylabel("Novelty Metric [%]")
fig.legend(ncol=3, loc='outside upper right')

plt.show()