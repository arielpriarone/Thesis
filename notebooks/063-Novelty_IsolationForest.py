# %%
# this don't work well because the cluster have differnt sizes
from matplotlib import projections
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
import numpy as np
import matplotlib 
from matplotlib import cm
import src
import importlib
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
import datetime as dt
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


# %% load the data
X = train_matrix # data to fit in the model
print(np.shape(X))


# %% fit model 
clf = IsolationForest(random_state=0,verbose=True,max_samples=np.shape(X)[0])
clf.fit(X)

# %% predict with all dataset
threshold = 0.07
scores = clf.decision_function(test_matrix)
metric = - scores

fig, ax = plt.subplots()
ax.scatter(timestamps_test,metric,c='k',marker='.', s=2, label='Novelty metric')
ax.set_xlabel('timestamp')
ax.set_ylabel('metric')
ax.axhline(y=threshold, color='k', linestyle='-.', label='threshold')
ax.annotate('Novel behaviour\n2003-11-22 10:16', xy = (dt.datetime.fromisoformat('2003-11-22T10.16'), threshold), 
             fontsize = 12, xytext = (dt.datetime.fromisoformat('2003-11-13T15.06'), 0.15), 
             arrowprops = dict(facecolor = 'k', arrowstyle = '->'),
             color = 'k')
ax.legend()



plt.show()