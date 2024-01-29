# %%
import importlib
import time
from bson import Timestamp
import matplotlib # COMMENTO AGGIUNTO
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
import matplotlib
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

sensors = ["Bearing 3 x"]#["Bearing 1 x" , "Bearing 1 y", "Bearing 2 x", "Bearing 2 y", "Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"]

# RMS
def rms(x):
    return np.sqrt(np.mean(np.array(x)**2))
def normalize(x):
    std = StandardScaler()
    return std.fit_transform(x.reshape(-1,1)).flatten()

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
data = {"$\mu$": mu, "RMS": RMS, "P2P": p2p, "$\hat{\sigma}$": std, "$\hat{\gamma}$": skew, "$\hat{\kappa}$": kurtosis}
# %% Plot

fig, ax = plt.subplots(6,1,sharex=True)
for i, metric in enumerate(["$\mu$", "RMS", "P2P", "$\hat{\sigma}$", "$\hat{\gamma}$", "$\hat{\kappa}$"]):
    for sensor in sensors:
        ax[i].plot(timestamps,normalize(data[metric][sensor]), label=sensor, color='k', linewidth=0.5)
        ax[i].set_ylabel(metric)
        
ax[i].set_xlabel('timestamp')
ax[i].tick_params(axis='x', rotation=45)
ax[2].annotate("Novel behaviour\n2003-11-22 16:06", (datetime.fromisoformat("2003-11-22 16:06"), 1.4),(timestamps[800], 3.5), arrowprops=dict(arrowstyle="->"), fontsize=9)
#ax[0].legend(ncol=4, bbox_to_anchor=(0.5, 1.4), loc='upper center', fontsize=8)


plt.tight_layout()
plt.show()

# %%
