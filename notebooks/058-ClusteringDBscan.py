# %%
from cProfile import label
from gc import collect
from tkinter import font
from xml.dom.minidom import Document
import arrow
from click import style
from matplotlib import projections
from matplotlib.patches import ArrowStyle
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
import numpy as np
import matplotlib 
from matplotlib import cm
from sklearn.model_selection import train_test_split
import src
import importlib
import pickle 
import os
from sklearn.metrics import silhouette_score, silhouette_samples
from rich import print
from sklearn.cluster import DBSCAN
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import datetime as dt
from src.data.data import DB_Manager, mongoConnect
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.vis.set_matplotlib_params()
def dbscan_predict(model, X):

    nr_samples = X.shape[0]

    y_new = np.ones(shape=nr_samples, dtype=int) * -1
    dist_array = np.array([])
    for i in range(nr_samples):
        diff = model.components_ - X[i, :]  # NumPy broadcasting

        dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

        shortest_dist_idx = np.argmin(dist)
        dist_array = np.append(dist_array, dist[shortest_dist_idx])
        if dist[shortest_dist_idx] < model.eps: # check if distance is smaller than epsilon
            y_new[i] = model.labels_[model.core_sample_indices_[shortest_dist_idx]]

    return y_new, dist_array

# script settings
# Connect to the MongoDB database
client = MongoClient()

# Access the "BACKUP" database
database = client["BACKUP"]

# Access the "500_samples_OnlyBearing3x_allfeatures" collection
collection = database["500_samples_OnlyBearing3x_allfeatures"]

# Find the document with id "evaluated dataset"
test_dataset = collection.find_one({"_id": "evaluated dataset"})
train_dataset = collection.find_one({"_id": "training set"})
print(f"test dataset keys:\n {test_dataset.keys()}")
print(f"train dataset keys:\n {train_dataset.keys()}")
print(f"there are {len(train_dataset['Bearing 3 x'].keys())} features in the test dataset")

timestamps_train = list(train_dataset['timestamp'])
timestamps_test = list(test_dataset['timestamp'])
test_matrix = np.array(list(test_dataset['Bearing 3 x'].values())).transpose()
train_matrix = np.array(list(train_dataset['Bearing 3 x'].values())).transpose()

print(f"test matrix shape: {test_matrix.shape}")
print(f"train matrix shape: {train_matrix.shape}")

std_scaler = StandardScaler()
train_matrix = std_scaler.fit_transform(train_matrix)
test_matrix = std_scaler.transform(test_matrix)

# %% training

n_labels = []
sil_score = []
inertia = []
eps_range = np.linspace(1,20,100)
for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=1).fit(train_matrix)
    labels = [x for x in db.labels_ if x >=0] # drop the noise samples
    n_labels.append(len(np.unique(labels)))
    try:
        sil_score.append(silhouette_score(train_matrix, db.labels_))
    except:
        sil_score.append(0)


fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(eps_range,n_labels, 'k')
ax[0].set_ylabel('$n$ of sclusters')
ax[1].plot(eps_range,sil_score, 'k')
ax[1].set_ylabel('silhouette score')
ax[1].set_xlabel('$\epsilon$')
ax[1].annotate('optimal $\epsilon$ = 8\ngenerate 2 clusters', xy=(8, 0.558), xytext=(15, 0.2),arrowprops=dict(arrowstyle="->"))


# chose eps = 8
db = DBSCAN(eps=8, min_samples=1).fit(train_matrix)
    

# predict only train dataset
predictions_train_lab, predictions_train_dist = dbscan_predict(db,train_matrix)

print(predictions_train_lab, predictions_train_dist)


# %% predict with test dataset

predictions_test_lab, predictions_test_dist = dbscan_predict(db,train_matrix)

print(predictions_test_lab, predictions_test_dist)

# %% predict test dataset

predictions_lab, predictions_dist = dbscan_predict(db,test_matrix)

print(predictions_lab, predictions_dist)

# %% plot
fig, ax = plt.subplots()
threshold = 0.6
cmap = cm.get_cmap("Set1")
#ax.scatter(range(len(predictions_dist)),predictions_dist,c=[cmap(x) for x in predictions_lab],marker='.',s=.5)
ax.scatter(timestamps_test,predictions_dist,c='k',marker='.',s=1, label= 'Novelty metric value')
ax.axhline(y=db.eps*(1+threshold), color='k', linestyle='-.', label=f'novelty threshold = $\epsilon$*(1+{threshold})')
ax.legend()
ax.annotate('Novel behaviour\n2003-11-22 15:06', xy = (dt.datetime.fromisoformat('2003-11-22T15.06'), 24), 
             fontsize = 12, xytext = (dt.datetime.fromisoformat('2003-11-13T15.06'), 200), 
             arrowprops = dict(facecolor = 'k', arrowstyle = '->'),
             color = 'k')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Novelty metric')

# %% if runas script show plots
if not auxpath=='.':
    plt.tight_layout()
    plt.show()