from calendar import c
from itertools import count
from operator import le
from os import times
from pathlib import Path
from tracemalloc import stop
from matplotlib import lines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from rich import print
import src
from sklearn.preprocessing import StandardScaler
from matplotlib import ticker
src.visualization.set_matplotlib_params()

from pandas.plotting import scatter_matrix

filepath_test = r"data\processed\Shaker_test02\test_data_shaker.csv"
filepath_train = r"data\processed\Shaker_test02\train_data_shaker.csv"
tests_separators = [10,10,10,10,10,10,10,10,10,10,10]; # number of samples per test
separator_index = [0] + list(np.cumsum(tests_separators)) # index of the start of each test
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
# model of the micro
model = {
    'n_clusters': 4,
    'centers': [
        [0.7107763495560776, 0.13504565182945946, -0.8775567952867753, 0.4311111009431372, -0.9116528401783603, -0.7639974849126245, -0.8826249186556334],
        [-1.703008731888498, -1.4734313674427566, 1.5333927526890774, 0.18434453893877897, 1.4180484963527293, 0.8985265324389914, 1.4922919056340718],
        [0.2887162150172579, 1.2046332459821292, 0.21272680742067332, 0.2408753393311418, 0.3974166413880705, 0.21138671298908396, 0.25709648810500824],
        [-1.7032627631307642, -1.4733359102427546, 1.5337568568739655, -1.9542070763058088, 1.41592185337629, 1.582596115019071, 1.504017229610121]
    ],
    'stds': [72.95143809510259, 575891341.8682002, 200578.32468086746, 7940.905464949911, 15427.628903013298, 1915.0306669815125, 4594.875000274002],
    'means': [1847.6834319131162, 22001878406.922306, 247723.75760067653, 157839.1469004324, 32169.004794538432, 2722.5224566287525, 8479.908734171591],
    'radiuses': [1.2828838566475935, 1.143139176317999, 1.9723564304733, 1.7256753322678466],
    'weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}



# %% Load the data
data = pd.read_csv(filepath_test, sep='\t')
print(data.keys())  
print(data.head())

train_data = pd.read_csv(filepath_train, sep='\t')

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
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_xlabel("Sample [-]")
ax.set_ylabel("Novelty Metric [%]")
ax.set_xlabel("Sample [-]")
ax.set_ylabel("Novelty Metric [%]")
fig.legend(ncol=3, loc='outside upper right')

# plot the features in s confusion matrix
features = train_data.keys()[1:-1]

features_matrix = train_data[features].to_numpy()

stdsclr=StandardScaler()
features_matrix = stdsclr.fit_transform(features_matrix)
features_matrix_test = stdsclr.transform(data[features].to_numpy())

print(features)

fig, ax = plt.subplots(len(features),len(features), figsize=(5.78851,7.18))
for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        if i==j:
            ax[i,j].hist(features_matrix[:,j], bins='auto', color='k')
        else:
            ax[i,j].scatter(features_matrix[:,j],features_matrix[:,i], c='k', s=2, marker='.')
            ax[i,j].scatter(features_matrix_test[separator_index[6]:separator_index[7],j],features_matrix_test[separator_index[6]:separator_index[7],i], c='r', s=2, marker='.')
            ax[i,j].scatter(features_matrix_test[separator_index[8]:separator_index[9],j],features_matrix_test[separator_index[8]:separator_index[9],i], c='m', s=2, marker='.')
            for cluster in range(model['n_clusters']):
                ax[i,j].add_patch(plt.Circle((model['centers'][cluster][j],model['centers'][cluster][i]),model['radiuses'][cluster],color='gray',fill=True, alpha=0.2, linestyle=''))
        ax[i,j].xaxis.set_tick_params(labelbottom=False)
        ax[i,j].yaxis.set_tick_params(labelleft=False)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

        if i==len(features)-1:
            ax[i,j].set_xlabel(feature2)

        if j==0:
            ax[i,j].set_ylabel(feature1)

fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
            
# try with lof
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(features_matrix)
NoveltyMetric = -lof.decision_function(features_matrix_test)

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

fig, ax = plt.subplots()
start = 1
for i, test in enumerate(tests):
    xlabels = [dt.datetime.fromisoformat(ts) for ts in timestamps[i]]
    ax.plot(range(start,start+len(test)),test*100, label=tests_names[i])
    start += len(test)
    print(f"Test {i} - {tests_names[i]}")
    print(f"Novelty: {(test)}")
ax.axhline(y=0, color='k', linestyle='-.', label='threshold')
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
ax.set_xlabel("Sample [-]")
ax.set_ylabel("Novelty Metric [%]")
fig.legend(ncol=3, bbox_to_anchor=(0.5, 1.02), loc='upper center')

plt.show()