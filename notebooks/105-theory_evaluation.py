# %%

from cProfile import label
from calendar import c
from hmac import new
from turtle import color
from matplotlib.patches import Shadow
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyparsing import col, line
from pyrsistent import b
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import src
from matplotlib.pyplot import cm # type: ignore
from matplotlib.colors import LightSource
import matplotlib as mpl 

rc_fonts = {
    "text.usetex": True,
    #'text.latex.preview': True, # Gives correct legend alignment.
    'mathtext.default': 'regular',
}
mpl.rcParams.update(rc_fonts)
plt.rc('text.latex', preamble=r'\usepackage{bm}')
import matplotlib.pylab as plt


def plot_sphere(ax, radius, center, color='b', alpha=0.1):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color=color, linewidth=0.1, zorder=1, linestyle='solid', antialiased=True)

src.vis.set_matplotlib_params()

# Generate random blobs
n_clusters = 2
X, y = make_blobs(n_samples=400, centers=n_clusters, n_features=3, random_state=1) # type: ignore

# Apply k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# Separate the clusters
cluster1 = X[kmeans.labels_ == 0]
cluster2 = X[kmeans.labels_ == 1]

# Plot the blobs and cluster centers
cmap = cm.tab10
colors = [cmap(i) for i in kmeans.labels_]
print(cmap)
print(kmeans.labels_)


fig = plt.figure()
plt.subplots_adjust(top=1.0,
bottom=0.045,
left=0.0,
right=0.709,
hspace=0.2,
wspace=0.2)
ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
ax.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], color=cmap(0), marker='.', label=r'$\bm{\mathcal{C}}_i$',zorder=2)
ax.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], color=cmap(1), marker='.', label=r'$\bm{\mathcal{C}}_j$',zorder=2)
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],s=50, c='k', marker='x', label=r'$\bm{c_i}$, $\bm{c}_j$',zorder=2)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')

# Calculate the radius of each cluster
radii = []
for i in range(n_clusters):
    cluster_points = X[kmeans.labels_ == i]
    centroid = kmeans.cluster_centers_[i]
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    radius = np.max(distances)
    radii.append(radius)
    print(f"Cluster {i+1} radius: {radius}")

# Plot the smooth transparent surfaces
for i in range(n_clusters):
    plot_sphere(ax, radii[i], kmeans.cluster_centers_[i], color=cmap(i), alpha=0.1)



# Generate a new random instance
new_instance = np.array([-4,12,-6])

# Assign the new instance to a cluster
new_instance_cluster = kmeans.predict([new_instance])[0]
ax.scatter(new_instance[0], new_instance[1], new_instance[2], c='magenta', marker='.', s=100, label=r'$\bm{\mathcal{S}}_n$',zorder=5)

# Plot an arrow from the center of the cluster to the new instance
length = np.linalg.norm(new_instance - kmeans.cluster_centers_[new_instance_cluster])
ax.quiver(kmeans.cluster_centers_[new_instance_cluster][0], kmeans.cluster_centers_[new_instance_cluster][1], kmeans.cluster_centers_[new_instance_cluster][2],
          new_instance[0] - kmeans.cluster_centers_[new_instance_cluster][0], new_instance[1] - kmeans.cluster_centers_[new_instance_cluster][1], new_instance[2] - kmeans.cluster_centers_[new_instance_cluster][2],
          color=cmap(0), arrow_length_ratio=1.5/length, label=r'$\bm{d}_{n,i}$',zorder=4)

new_instance_cluster = 0 if new_instance_cluster == 1 else 1 # consider now the disccarded cluster
length = np.linalg.norm(new_instance - kmeans.cluster_centers_[new_instance_cluster])
ax.quiver(kmeans.cluster_centers_[new_instance_cluster][0], kmeans.cluster_centers_[new_instance_cluster][1], kmeans.cluster_centers_[new_instance_cluster][2],
          new_instance[0] - kmeans.cluster_centers_[new_instance_cluster][0], new_instance[1] - kmeans.cluster_centers_[new_instance_cluster][1], new_instance[2] - kmeans.cluster_centers_[new_instance_cluster][2],
          color=cmap(1), arrow_length_ratio=1.5/length, label=r'$\bm{d}_{n,j}$',zorder=4)

# Find the instance in the first cluster that is farthest from the center
cluster_points = X[kmeans.labels_ == 0]
centroid = kmeans.cluster_centers_[0]
distances = np.linalg.norm(cluster_points - centroid, axis=1)
farthest_instance_index = np.argmax(distances)
farthest_instance = cluster_points[farthest_instance_index]

# Plot an arrow from the center to the farthest instance
length = np.linalg.norm(farthest_instance - kmeans.cluster_centers_[0])
ax.quiver(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2],
          farthest_instance[0] - kmeans.cluster_centers_[0][0], farthest_instance[1] - kmeans.cluster_centers_[0][1], farthest_instance[2] - kmeans.cluster_centers_[0][2],
          color="blue", arrow_length_ratio=1.5/length, label=r'$\bm{r}_i$',zorder=4)

# Find the instance in the first cluster that is farthest from the center
cluster_points = X[kmeans.labels_ == 1]
centroid = kmeans.cluster_centers_[1]
distances = np.linalg.norm(cluster_points - centroid, axis=1)
farthest_instance_index = np.argmax(distances)
farthest_instance = cluster_points[farthest_instance_index]

# Plot an arrow from the center to the farthest instance
length = np.linalg.norm(farthest_instance - kmeans.cluster_centers_[1])
ax.quiver(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2],
          farthest_instance[0] - kmeans.cluster_centers_[1][0], farthest_instance[1] - kmeans.cluster_centers_[1][1], farthest_instance[2] - kmeans.cluster_centers_[1][2],
          color="red", arrow_length_ratio=1.5/length, label=r'$\bm{r}_j$',zorder=4)

ax.legend(loc='right',bbox_to_anchor=(1.7, 0.5)) 

ax.set_aspect('equal')
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.show()
