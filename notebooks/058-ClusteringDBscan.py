# %%
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
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                             # empty dictionary to save data 

filehandler = open(savepath, 'rb') 
IMSDATA = pickle.load(filehandler)
print(IMSDATA.keys())


# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(IMSDATA['wavanaly_standardized_train'][:,10],IMSDATA['wavanaly_standardized_train'][:,11],IMSDATA['wavanaly_standardized_train'][:,12],s=1,marker='.',c='black')

# %% training
print(IMSDATA['wavanaly_standardized_train'].shape)

n_labels = []
sil_score = []
inertia = []
eps_range = np.linspace(1,20,100)
for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=1).fit(IMSDATA['wavanaly_standardized_train'])
    labels = [x for x in db.labels_ if x >=0] # drop the noise samples
    n_labels.append(len(np.unique(labels)))
    try:
        sil_score.append(silhouette_score(IMSDATA['wavanaly_standardized_train'], db.labels_))
    except:
        sil_score.append(0)


fig, ax = plt.subplots(2,1,sharex=True)
ax[0].scatter(eps_range,n_labels)
ax[0].set_ylabel('n sclusters')
ax[1].scatter(eps_range,sil_score)
ax[1].set_ylabel('silhouette score')
ax[1].set_xlabel('eps')

# chose eps = 8
db = DBSCAN(eps=8, min_samples=1).fit(IMSDATA['wavanaly_standardized_train'])
    

# predict only train dataset
predictions_train_lab, predictions_train_dist = dbscan_predict(db,IMSDATA['wavanaly_standardized_train'])

print(predictions_train_lab, predictions_train_dist)


# %% predict with test dataset

predictions_test_lab, predictions_test_dist = dbscan_predict(db,IMSDATA['wavanaly_standardized_test'])

print(predictions_test_lab, predictions_test_dist)

# %% predict with all dataset

predictions_lab, predictions_dist = dbscan_predict(db,IMSDATA['wavanaly_standardized'])

print(predictions_lab, predictions_dist)

# %% plot
fig, ax = plt.subplots()
threshold = 0.4
cmap = cm.get_cmap("Set1")
ax.scatter(range(len(predictions_dist)),predictions_dist,c=[cmap(x) for x in predictions_lab],marker='.')
ax.axhline(y=db.eps*(1+threshold), color='r', linestyle='-.')
ax.set_xlabel('sample')
ax.set_ylabel('distance')

# %% if runas script show plots
if not auxpath=='.':
    plt.tight_layout()
    plt.show()