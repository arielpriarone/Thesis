import numpy as np
import matplotlib.pyplot as plt
from rich import print

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# generate three blobs
X, y = make_blobs(n_samples=400, centers=3, n_features=2, random_state=1)

# plot the blobls
fig, ax = plt.subplots()
# generate a colormap with two colors defined as hex
cmap = plt.cm.colors.ListedColormap(['#002B49', '#EF7B00', '#FFC72C'])
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='.', s=10)

ax.set_aspect('equal')
ax.set_xlabel('X[0]')
ax.set_ylabel('X[1]')

plt.show()