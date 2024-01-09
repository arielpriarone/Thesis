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

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(IMSDATA['wavanaly_standardized_train'][:,10],IMSDATA['wavanaly_standardized_train'][:,11],IMSDATA['wavanaly_standardized_train'][:,12],s=1,marker='.',c='black')

# %% training
X = IMSDATA['wavanaly_standardized_train'] # data to fit in the model
print(np.shape(X))

# %% fit model 
GM = BayesianGaussianMixture(n_components=np.shape(X)[0], covariance_type='full', random_state=0)
GM.fit(X)
print(f'The mixture model has converged: {GM.converged_}, with {GM.n_iter_} iterations')

# %% predict with test dataset

scores = GM.score_samples(IMSDATA['wavanaly_standardized_test'])
indicator = -scores
predictions = GM.predict(IMSDATA['wavanaly_standardized_test'])

fig, ax = plt.subplots()
ax.scatter(range(len(indicator)),indicator,c=predictions)
ax.set_xlabel('sample')
ax.set_ylabel('density')


# %% predict with all dataset
threshold = 0.005
scores = GM.score_samples(IMSDATA['wavanaly_standardized'])
indicator = -scores
predictions = GM.predict(IMSDATA['wavanaly_standardized'])

fig, ax = plt.subplots()
ax.scatter(range(len(indicator)),indicator,c=predictions,marker='.')
ax.set_xlabel('sample')
ax.set_ylabel('density')
ax.axhline(y=threshold, color='r', linestyle='-.')
# ax.set_yscale('log')


# check on second dataset
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized_second.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                              # empty dictionary to save data 
filehandler = open(savepath, 'rb') 
IMSDATA = pickle.load(filehandler)

scores = GM.score_samples(IMSDATA['wavanaly_standardized'])
indicator = -scores
predictions = GM.predict(IMSDATA['wavanaly_standardized'])

fig, ax = plt.subplots()
ax.scatter(range(len(indicator)),indicator,c=predictions)
ax.set_xlabel('sample')
ax.set_ylabel('density')
ax.set_yscale('log')


# %% if runas script show plots
if not auxpath=='.':
    plt.tight_layout()
    plt.show()