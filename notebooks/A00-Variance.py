from calendar import c
from operator import le
from statistics import variance
from matplotlib import axis
import numpy as np
import matplotlib.pyplot as plt
from rich import print
import matplotlib as mpl

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import src

src.vis.set_matplotlib_params()

# generate three blobs
centers=[(1,1), (12,16), (27,2)]
X, y = make_blobs(n_samples=600, n_features=2, random_state=2, cluster_std=[1.5,3,1.5], centers=centers)
X=np.append(X,np.array([19,1]).reshape(-1,2), axis=0) # add an outlier
y=np.append(y,np.max(y)) # add an outlier label
print(f"X.shape = {X.shape}")
print(y.shape)
print(y)
# plot the blobls

colorlist = ["#ebac23", "#b80058", "#008cf9", "#006e00", "#00bbad", "#d163e6", "#b24502", "#ff9287", "#5954d6", "#00c6f8", "#878500", "#00a76c", "#bdbdbd"]
cmap=mpl.colors.ListedColormap(colorlist[0:len(np.unique(y))])
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='.', s=10)

# plot the centers
for center in centers:
    ax.scatter(center[0], center[1], c='black', marker='x', s=100)
# compute the radiuses
radiuses = dict()
stds = dict()
for center,y_int in zip(centers, np.arange(len(centers))): # for each cluster
    current_pts = X[y==y_int, :]
    N = current_pts.shape[0]
    print(f"Y_int={y_int}, \t current data = {current_pts}")
    dists = np.linalg.norm(current_pts-center, axis=1,ord=2)
    radiuses[y_int] = np.max(dists)
    stds[y_int] = np.sqrt(np.sum((np.linalg.norm(current_pts-center,ord=2))**2)/N)
print(radiuses)
print(stds)

# plot the radiuses and stds
for center,y_int in zip(centers, np.unique(y)):
    ax.add_patch(plt.Circle(center, stds[y_int], fill=False, color='k', linestyle='--', linewidth=1))
    ax.add_patch(plt.Circle(center, radiuses[y_int], fill=False, color='k', linestyle='-', linewidth=1))

ax.set_aspect('equal')
ax.set_xlabel('F_1')
ax.set_ylabel('F_2')

plt.show()