# %%
import importlib # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
import src
import importlib
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileNames= ['2003.10.22.12.06.24', '2003.11.25.23.39.56']

snap=src.data.snapshot()
samplFreq=20000 #hz

# Plot
fig, ax = plt.subplots(2,1, sharex=True)
snap.readImsFile(path=dirPath+fileNames[0], variables="Bearing 3 x")
ax[0].plot(snap.rawData["time"], snap.rawData["Bearing 3 x"], label = "healthy", linewidth=0.1)
snap.readImsFile(path=dirPath+fileNames[1], variables="Bearing 3 x")
ax[1].plot(snap.rawData["time"], snap.rawData["Bearing 3 x"], label = "healthy", linewidth=0.1)
ax[1].set_xlabel('time [s]')
ax[0].set_ylabel('amplitude')
ax[1].set_ylabel('amplitude')


plt.tight_layout()
plt.show()
