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
from sklearn.metrics import silhouette_score, silhouette_samples

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
db = DBSCAN(eps=0.78, min_samples=10).fit(X_aniso)
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
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(handles=legend_elements, loc='center left',bbox_to_anchor=(1, 0.5))
fig.tight_layout()

bbox = ax.get_position()

# plot data
fig, ax = plt.subplots()
ax.scatter(X_aniso[:, 0], X_aniso[:, 1],marker=".", c='k',label='datapoints')
ax.legend(loc='center left',bbox_to_anchor=(1, 0.5))
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
fig.tight_layout()
ax.set_position(bbox)

# plot clustering with silhouette
n_labels = []
sil_score = []
inertia = []
eps_range = np.linspace(0.3,1,100)
for eps in eps_range:
    db = DBSCAN(eps=eps, min_samples=10).fit(X_aniso)
    labels = [x for x in db.labels_ if x >=0] # drop the noise samples
    n_labels.append(len(np.unique(labels))) # only count clusters
    try:
        sil_score.append(silhouette_score(X_aniso, db.labels_))
    except:
        sil_score.append(0)
src.vis.set_matplotlib_params() # reset the values
fig, ax = plt.subplots(2,1,sharex=True)
ax[0].plot(eps_range,n_labels)
ax[0].set_ylabel('$n$ of sclusters')
ax[1].plot(eps_range,sil_score)
ax[1].set_ylabel('silhouette score')
ax[1].set_xlabel(r'$\varepsilon$')

ax[0].annotate(r'coresponding to $3$ clusters', xy=(0.78, 3), xytext=(0.6, 1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )
ax[1].annotate(r'max for $\varepsilon \approx 0.78$', xy=(0.78, 0.55), xytext=(0.6, 0.3),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )

fig.tight_layout()


plt.show()