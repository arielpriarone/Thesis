from cProfile import label
import matplotlib
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import src
from sklearn import metrics
from sklearn.cluster import DBSCAN
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

# settings
src.vis.set_matplotlib_params()
matplotlib.rcParams['figure.figsize'] = (matplotlib.rcParams['figure.figsize'][0], matplotlib.rcParams['figure.figsize'][1]*0.6) # set size of plots
n_samples = 120

# Anisotropicly distributed data
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170, centers= np.array([[10, 10], [-10, -10]]))
transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_aniso = np.dot(X, transformation)
X, y = datasets.make_blobs(n_samples=int(n_samples/2), random_state=170, centers=np.array([[0,0]]))
transformation = np.dot(transformation, np.array([[0, -1],[1,  0]]))
scale=0.6
transformation = np.dot(transformation, np.array([[scale, 0],[0,  scale]]))
X_aniso = np.concatenate((X_aniso, np.dot(X, transformation)), axis = 0)

# clustewring
db = DBSCAN(eps=0.7, min_samples=10).fit(X_aniso)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

# plot clustering
fig, ax = plt.subplots()
unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

cmap = plt.get_cmap('Set1')
colors = [cmap(each) for each in np.linspace(0, 1, len(unique_labels))]
print(colors)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = X_aniso[class_member_mask & core_samples_mask]
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        marker="*",
        color=tuple(col)
    )

    xy = X_aniso[class_member_mask & ~core_samples_mask]
    if k == -1:
        # Black used for noise.
        marker = "x"
    else:
        marker = "."
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        marker=marker,
        color=tuple(col)
    )
legend_elements = [
    Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=10, label='Core Points'),
    Line2D([0], [0], marker='.', color='w', markerfacecolor='black', markersize=10, label='Reachable Points'),
    Line2D([0], [0], marker='x', linestyle='', color='k', markerfacecolor='black', markersize=5, label='Outliers'),    
    Patch(color=colors[1], label='Clust. 1'),
    Patch(color=colors[2], label='Clust. 2'),
    Patch(color=colors[3], label='Clust. 3'),
]
ax.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1, 0.5))
fig.tight_layout()

bbox = ax.get_position()

# plot data
fig, ax = plt.subplots()
ax.scatter(X_aniso[:, 0], X_aniso[:, 1],marker=".", c='k',label='datapoints')
ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
fig.tight_layout()
ax.set_position(bbox)



plt.show()