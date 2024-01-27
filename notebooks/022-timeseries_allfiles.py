# %%
import importlib
import time
from bson import Timestamp # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
import src
import os
import importlib
from rich import print
from datetime import datetime
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

sensors = ["Bearing 1 x", "Bearing 1 y", "Bearing 2 x"]

# RMS
def rms(x):
    return np.sqrt(np.mean(x**2))

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileNames = os.listdir(dirPath)
snap=src.data.snapshot()

RMS= {}
for sensor in sensors:
    RMS[sensor] = np.array([])
timestamps = []

for i, fileName in enumerate(fileNames):
    # check if current path is a file
    if os.path.isfile(os.path.join(dirPath, fileName)):
        print(f"Reading file {i+1}/{len(fileNames)}: {fileName}")
        snap.readImsFile(path=dirPath+fileName, variables=sensors)
        for sensor in sensors:
            RMS[sensor] = np.append(RMS[sensor], rms(snap.rawData[sensor]))
        timestamps.append(src.data.IMS_filepathToTimestamp(fileName))

# Plot
fig, ax = plt.subplots(1,1)
for sensor in sensors:
    ax.plot(timestamps,RMS[sensor], label=sensor)
ax.set_xlabel('sample')
ax.set_ylabel('RMS')


plt.tight_layout()
plt.show()
