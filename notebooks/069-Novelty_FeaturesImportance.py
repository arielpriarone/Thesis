# %%
# this don't work well because the cluster have differnt sizes
from matplotlib import projections
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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

# train a kmeans with different cluster number
sil_score=[]
range_clusters = range(2,10)
for n_clust in range_clusters:
    kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=10).fit(X)
    sil_score.append(silhouette_score(X, kmeans.labels_))
    print("n_clust:",n_clust,"silhouette_score:",silhouette_score(X, kmeans.labels_))

# %% plot the silhouette score
fig, ax = plt.subplots()
ax.plot(range_clusters,sil_score,c='k',marker='.')

# %% select the best number of clusters
n_clust = range_clusters[np.argmax(sil_score)] # selecting the max number of clusters
kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=10).fit(X)
print("kmeans trained with n_clust:",n_clust)

# %% fit model 
RF = RandomForestClassifier(random_state=0,verbose=True)
RF.fit(X,kmeans.labels_)

# verify accuracy of classificatoin
correct_predictions = RF.predict(X) == kmeans.labels_
print("Accuracy:",np.sum(correct_predictions)/len(correct_predictions))

# %% select important features
fig, ax = plt.subplots()
ax.bar(range(len(RF.feature_importances_)),RF.feature_importances_,color='k')
ax.set_xlabel('feature')
ax.set_ylabel('importance')
ax.set_xticks(range(len(RF.feature_importances_)))
plt.title("Feature importance")


plt.show()