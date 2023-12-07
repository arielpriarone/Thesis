# %% include
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

# %% read the first test results

path = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\test_first_wave.log"
errors_wave1 = [] # list of errors - prediction metric

with open(path, "r") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == "End of features.":
            try:
                number = float(lines[i+1].strip())
                errors_wave1.append(number*100)
            except ValueError:
                raise ValueError(f"line {i+1} is not a number")

print(f"shape of the rerror, first wave: {np.shape(errors_wave1)}")

# %% read the second test results

path = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\test_second_wave.log"
errors_wave2 = [] # list of errors - prediction metric

with open(path, "r") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == "End of features.":
            try:
                number = float(lines[i+1].strip())
                errors_wave2.append(number*100)
            except ValueError:
                raise ValueError(f"line {i+1} is not a number")

print(f"shape of the rerror, second wave: {np.shape(errors_wave2)}")


# %% read the novelty test results

path = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\test_noveltydetected.log"
errors_novelty = [] # list of errors - prediction metric

with open(path, "r") as file:
    lines = file.readlines()
    for i in range(len(lines)):
        if lines[i].strip() == "End of features.":
            try:
                number = float(lines[i+1].strip())
                errors_novelty.append(number*100)
            except ValueError:
                raise ValueError(f"line {i+1} is not a number")

print(f"shape of the rerror, second wave: {np.shape(errors_novelty)}")

# %% plot the first test results
fig, ax = plt.subplots()
ax.plot(range(1,len(errors_wave1)+1),errors_wave1)
ax.plot(range(len(errors_wave1)+1,len(errors_wave1)+len(errors_wave2)+1),errors_wave2)
ax.plot(range(len(errors_wave1)+len(errors_wave2)+1,len(errors_wave1)+len(errors_wave2)+len(errors_novelty)+1),errors_novelty)
ax.hlines(10,0,len(errors_wave1)+len(errors_wave2)+len(errors_novelty)+1,linestyles='dashed',colors=['red'])
ax.set_xlabel("Test number")
ax.set_ylabel("Relative distance to closest cluster [%]")
ax.legend(["First wave","Second wave","Armonic injection"])
ax.grid(visible=True,which='both')


plt.tight_layout()

plt.show()