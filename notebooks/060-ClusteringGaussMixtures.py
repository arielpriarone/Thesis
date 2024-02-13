# %%
from cProfile import label
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
from sklearn.mixture import GaussianMixture
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.vis.set_matplotlib_params()
import datetime as dt
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

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

max_clusters=30
BIC = [] # Bayesian Information Criterion
AIC = [] # Akaike Information Criterion
X = train_matrix # data to fit in the model
for n_blobs in range(1,max_clusters+1):
    GM = GaussianMixture(n_components=n_blobs, covariance_type='full', random_state=0)
    GM.fit(X)
    print(f'Number of clusters: {n_blobs}: the mixture model has converged: {GM.converged_}, with {GM.n_iter_} iterations')
    BIC.append(GM.bic(X))
    AIC.append(GM.aic(X))
# plot BIC and AIC
fig, ax = plt.subplots()
ax.plot(range(1,max_clusters+1),BIC,label='BIC', color='k', linestyle='--')
ax.plot(range(1,max_clusters+1),AIC,label='AIC', color='k', linestyle='dashdot')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Information Criterion')
# ax.set_yscale('symlog')
ax.annotate('optimal 25 clusters', xy=(25, -1e05), xytext=(15, 0),arrowprops=dict(arrowstyle="->"))

ax.legend()

# %% fit model with 30 clusters
GM = GaussianMixture(n_components=25, covariance_type='full', random_state=0)
GM.fit(X)
print(f'The mixture model has converged: {GM.converged_}, with {GM.n_iter_} iterations')

# %% predict with test dataset

scores = GM.score_samples(test_matrix)
indicator = -scores
predictions = GM.predict(test_matrix)
threshold = 3.2646e07
fig, ax = plt.subplots()
ax.scatter(timestamps_test,indicator,c=predictions, s=2, marker='.', cmap='viridis', label='density value')
ax.hlines(threshold,timestamps_test[0],timestamps_test[-1], color='r', linestyle='-.', label='threshold')
ax.legend()
ax.annotate('Novel behaviour\n2003-11-22 03:47', xy = (dt.datetime.fromisoformat('2003-11-22T03.47'), threshold), 
             fontsize = 12, xytext = (dt.datetime.fromisoformat('2003-11-13T15.06'), 100*threshold), 
             arrowprops = dict(facecolor = 'k', arrowstyle = '->'),
             color = 'k')
ax.set_yscale('log')
ax.set_xlabel('Timestamp')
ax.set_ylabel('density')


# %% predict with all dataset
# threshold = 0.005
# scores = GM.score_samples(np.concatenate((train_matrix,test_matrix),axis=0))
# indicator = -scores
# predictions = GM.predict(np.concatenate((train_matrix,test_matrix),axis=0))

# fig, ax = plt.subplots()
# ax.scatter(range(len(indicator)),indicator,c=predictions,marker='.')
# ax.set_xlabel('sample')
# ax.set_ylabel('density')
# ax.axhline(y=threshold, color='r', linestyle='-.')
# ax.set_yscale('log')


# check on second dataset
# savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized_second.pickle") #file to save the analisys
# decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
# IMSDATA={}                                              # empty dictionary to save data 
# filehandler = open(savepath, 'rb') 
# IMSDATA = pickle.load(filehandler)

# scores = GM.score_samples(IMSDATA['wavanaly_standardized'])
# indicator = -scores
# predictions = GM.predict(IMSDATA['wavanaly_standardized'])

# fig, ax = plt.subplots()
# ax.scatter(range(len(indicator)),indicator,c=predictions)
# ax.set_xlabel('sample')
# ax.set_ylabel('density')
# ax.set_yscale('log')


# %% if runas script show plots
if not auxpath=='.':
    plt.tight_layout()
    plt.show()