# %%
import importlib
import time
from bson import Timestamp # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import src
import os
import importlib
from rich import print
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import scipy
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

sensors = ["Bearing 3 x"]

# RMS
def rms(x):
    return np.sqrt(np.mean(np.array(x)**2))
def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# database connection
mongoclient, database, collection = src.data.mongoConnect("BACKUP", "RawData_1st_test_IMSBearing","mongodb://localhost:27017/")


# get all documents
documents = collection.find({})

RMS= {}
mu = {}
p2p = {}
std = {}
skew  = {}
kurtosis = {}

for sensor in sensors:
    mu[sensor] = np.array([])
    RMS[sensor] = np.array([])
    p2p[sensor] = np.array([])
    std[sensor] = np.array([])
    skew[sensor] = np.array([])
    kurtosis[sensor] = np.array([])
timestamps = []

n_documents = collection.count_documents({})

for i, document in enumerate(documents):
    # check if current path is a file
    print(f"Reading document {i+1}/{n_documents}: {document['timestamp']}")
    for sensor in sensors:
        RMS[sensor] = np.append(RMS[sensor], rms(document[sensor]))
        mu[sensor] = np.append(mu[sensor], np.mean(document[sensor]))
        p2p[sensor] = np.append(p2p[sensor], np.max(document[sensor])-np.min(document[sensor]))
        std[sensor] = np.append(std[sensor], np.std(document[sensor]))
        skew[sensor] = np.append(skew[sensor], scipy.stats.skew(document[sensor]))
        kurtosis[sensor] = np.append(kurtosis[sensor], scipy.stats.kurtosis(document[sensor]))
    timestamps.append(document["timestamp"])

# Plot
fig, ax = plt.subplots(1,1)
for sensor in sensors:
    ax.plot(timestamps,normalize(mu[sensor]), label="$\mu$")
    ax.plot(timestamps,normalize(RMS[sensor]), label="RMS")
    ax.plot(timestamps,normalize(p2p[sensor]), label="P2P")
    ax.plot(timestamps,normalize(std[sensor]), label="$\hat{\sigma}$")
    ax.plot(timestamps,normalize(skew[sensor]), label="$\hat{\gamma}$")
    ax.plot(timestamps,normalize(kurtosis[sensor]), label="$\hat{\kappa}$")

ax.set_xlabel('timestamp')
ax.set_ylabel('Normalized time domain\n features')
ax.legend()
ax.tick_params(axis='x', rotation=45)


plt.tight_layout()
plt.show()
