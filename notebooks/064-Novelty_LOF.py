# %%
from math import inf
from matplotlib import projections
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
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
from sklearn.mixture import BayesianGaussianMixture
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.vis.set_matplotlib_params()

# script settings
modelpath     = auxpath + "./models/kmeans_model.pickle"   # folder path
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                             # empty dictionary to save data 

filehandler = open(savepath, 'rb') 
IMSDATA = pickle.load(filehandler)
print(IMSDATA.keys())


# %% load the data
X = IMSDATA['wavanaly_standardized_train'] # data to fit in the model
print(np.shape(X))
X_full = IMSDATA['wavanaly_standardized'] # data to fit in the model
print(np.shape(X_full))

# %% fit model 
lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.005)
lof.fit(X)

# %% predict with all dataset
threshold = 0.005
scores = lof.decision_function(IMSDATA['wavanaly_standardized'])
metric = - scores

fig, ax = plt.subplots()
ax.scatter(range(len(metric)),metric,c='k',marker='.')
ax.set_xlabel('snapshot')
ax.set_ylabel('LOF')
ax.axhline(y=threshold, color='r', linestyle='-.')

plt.show()