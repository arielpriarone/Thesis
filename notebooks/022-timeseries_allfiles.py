# %%
import importlib # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
import src
import os
import importlib
from rich import print
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

# RMS
def rms(x):
    return np.sqrt(np.mean(x**2))

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileNames = os.listdir(dirPath)
snap=src.data.snapshot()

RMS= []

for i, fileName in enumerate(fileNames):
    # check if current path is a file
    if os.path.isfile(os.path.join(dirPath, fileName)):
        print(f"Reading file {i+1}/{len(fileNames)}: {fileName}")
        snap.readImsFile(path=dirPath+fileName, variables="Bearing 3 x")
        RMS.append(rms(snap.rawData["Bearing 3 x"]))

# Plot
fig, ax = plt.subplots(1,1)
ax.plot(RMS)
ax.set_xlabel('sample')
ax.set_ylabel('RMS')


plt.tight_layout()
plt.show()
