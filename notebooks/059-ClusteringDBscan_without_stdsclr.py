# %%
from operator import is_
from matplotlib import projections
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
import numpy as np
import matplotlib 
from matplotlib import cm
import src
import importlib
import pickle 
import os
from sklearn.metrics import silhouette_score, silhouette_samples
from rich import print
from sklearn.cluster import DBSCAN
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
modelpath     = auxpath + "./models/kmeans_model.pickle"   # folder path
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized_second.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                             # empty dictionary to save data 

filehandler = open(savepath, 'rb') 
IMSDATA = pickle.load(filehandler)
print(IMSDATA.keys())


# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(IMSDATA['wavanaly_train'][:,10],IMSDATA['wavanaly_standardized_train'][:,11],IMSDATA['wavanaly_standardized_train'][:,12],s=1,marker='.',c='black')

# %% training
print(IMSDATA['wavanaly_train'].shape)

n_labels = []
eps_range = np.linspace(1,20,100)
for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=1).fit(IMSDATA['wavanaly_train'])
    n_labels.append(len(np.unique(db.labels_)))

fig, ax = plt.subplots()
ax.scatter(eps_range,n_labels)
ax.set_xlabel('eps')
ax.set_ylabel('n_labels')

# chose eps = 1.3
db = DBSCAN(eps=1.3, min_samples=1).fit(IMSDATA['wavanaly_train'])

# predict only train dataset
predictions_train_lab, predictions_train_dist = dbscan_predict(db,IMSDATA['wavanaly_train'])
print(predictions_train_lab, predictions_train_dist)


# %% predict with test dataset
predictions_test_lab, predictions_test_dist = dbscan_predict(db,IMSDATA['wavanaly_test'])
print(predictions_test_lab, predictions_test_dist)

# %% predict with all dataset

predictions_lab, predictions_dist = dbscan_predict(db,IMSDATA['wavanaly'])
print(predictions_lab, predictions_dist)

# %% plot
fig, ax = plt.subplots()
threshold = 0.3
cmap = cm.get_cmap("Set1")
ax.scatter(range(len(predictions_test_dist)),predictions_test_dist,c=[cmap(x) for x in predictions_test_lab])
ax.axhline(y=db.eps*(1+threshold), color='r', linestyle='-.')
ax.xaxis.grid(True, which='both', linestyle='--')
ax.yaxis.grid(True, which='both', linestyle='--')
ax.set_xlabel('sample')
ax.set_ylabel('distance')

# %% if runas script show plots
if not auxpath=='.':
    plt.tight_layout()
    plt.show()