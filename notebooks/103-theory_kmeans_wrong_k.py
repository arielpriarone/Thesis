from ast import arg
from matplotlib import markers
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from pandas import pivot
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from matplotlib.pyplot import cm, xlim # type: ignore
import matplotlib as mpl
import src
src.vis.set_matplotlib_params()
mpl.rcParams["scatter.marker"] = '.'

def voronoi_finite_polygons_2d(vor, radius=1000):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

# Generate random blobs
X, _ = make_blobs(n_samples=200, centers=3, random_state=6)
x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])

x_domain = np.array([x_min, x_max])
y_domain = np.array([y_min, y_max])

# compute best k
# plot the silhouette score
from sklearn.metrics import silhouette_score
S = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, init='random', random_state=405, max_iter=100, n_init=1)
    kmeans.fit(X)
    S.append(silhouette_score(X, kmeans.labels_))

fig, ax = plt.subplots()
fig.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.2, wspace=0.2)
x,y = fig.get_size_inches()
fig.set_size_inches(x, y*0.5)

ax.plot(range(2,10), S, marker='o')
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Silhouette score")
ax.set_xlim((0.5, 9.5))
ax.quiver(3,0.725,-1,0.05, angles='xy', scale_units='xy', scale=1, color='k', pivot= 'tip')
ax.text(4.2,0.675, "Max Silhouette", ha='left', va='center')

# assign k
k = S.index(max(S)) + 2 # max silhouette score +2 because we start at 2 clusters

# Initialize the k-means algorithm
kmeans = KMeans(n_clusters=k, init='random', random_state=405, max_iter=1, n_init=1)

# Fit the data to the algorithm
kmeans.fit(X)

# vor = Voronoi(kmeans.cluster_centers_)
# regions, vertices = voronoi_finite_polygons_2d(vor)
# print(1.1*x_domain)
# Plot the data points and decision boundaries
fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
colormap = cm.tab10

# Plot the original data points
border = 1
axs[0,0].scatter(X[:, 0], X[:, 1], c='k')
axs[0,0].set_xlim([x_domain[0]-border, x_domain[1]+border])
axs[0,0].set_ylim([y_domain[0]-border, y_domain[1]+border])
# Plot the decision boundaries and centroids
axs[0,1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=colormap)
axs[0,1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='k')

# # Plot the Voronoi diagram
# i = 0
# for region in regions:
#     polygon = vertices[region]
#     axs[0,1].fill(*zip(*polygon), alpha=0.4, color=colormap(i))
#     i += 1
# axs[0,1].set_xlim(axs[0,0].get_xlim())
# axs[0,1].set_ylim(axs[0,0].get_ylim())


# vor = Voronoi(kmeans.cluster_centers_)
# regions, vertices = voronoi_finite_polygons_2d(vor)

## *** second iteration ***
# Initialize the k-means algorithm
kmeans = KMeans(n_clusters=k, init='random', random_state=405, max_iter=2, n_init=1)

# Fit the data to the algorithm
kmeans.fit(X)

# vor = Voronoi(kmeans.cluster_centers_)
# regions, vertices = voronoi_finite_polygons_2d(vor)

# Plot the decision boundaries and centroids
axs[1,0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=colormap)
axs[1,0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='k')

# # Plot the Voronoi diagram
# i = 0
# for region in regions:
#     polygon = vertices[region]
#     axs[1,0].fill(*zip(*polygon), alpha=0.4, color=colormap(i))
#     i += 1
# axs[1,0].set_xlim(axs[0,0].get_xlim())
# axs[1,0].set_ylim(axs[0,0].get_ylim())

## *** finished iteration ***
# Initialize the k-means algorithm
kmeans = KMeans(n_clusters=k, init='random', random_state=405, max_iter=100, n_init=1)

# Fit the data to the algorithm
kmeans.fit(X)

# vor = Voronoi(kmeans.cluster_centers_)
# regions, vertices = voronoi_finite_polygons_2d(vor)

# Plot the decision boundaries and centroids
axs[1,1].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap=colormap)
axs[1,1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='k')

# # Plot the Voronoi diagram
# i = 0
# for region in regions:
#     polygon = vertices[region]
#     axs[1,1].fill(*zip(*polygon), alpha=0.4, color=colormap(i))
#     i += 1
# axs[1,1].set_xlim(axs[0,0].get_xlim())
# axs[1,1].set_ylim(axs[0,0].get_ylim())

# configure plots
axs[0,0].set_title("Original data")
axs[0,1].set_title("Iteration 1")
axs[1,0].set_title("Iteration 2")
axs[1,1].set_title("Algorithm converged")

axs[0,0].set_ylabel("Feature 2")
axs[1,0].set_ylabel("Feature 2")
axs[1,0].set_xlabel("Feature 1")
axs[1,1].set_xlabel("Feature 1")


# plot the boudaries
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
fig, axs = plt.subplots(1,3, sharex=True, sharey=True)

new_snap = [3,-4]
axs[0].set_ylabel("Feature 2")
for k in range(2, 5):
    # Initialize the k-means algorithm
    kmeans = KMeans(n_clusters=k, max_iter=100, n_init=1)

    # Fit the data to the algorithm
    kmeans.fit(X)

    # compute the radiuses of the clusters
    radiuses = []
    for i in range(k):
        radiuses.append(np.linalg.norm(X[kmeans.labels_ == i] - kmeans.cluster_centers_[i], axis=1).max())

    axs[k-2].scatter(X[:, 0], X[:, 1], c='k', s=1)
    axs[k-2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='r')
    axs[k-2].set_xlabel("Feature 1")

    # Plot circles using the provided center coordinates and radii
    for center, radius in zip(kmeans.cluster_centers_, radiuses):
        circle = Circle(center, radius, edgecolor='k', facecolor='none')
        axs[k-2].add_patch(circle)

    axs[k-2].set_aspect('equal')

    # plot the new point
    axs[k-2].scatter(new_snap[0], new_snap[1], marker='o', color='magenta')

    # Create a custom legend with specified elements
    legend_elements = [Line2D([0], [0], marker='o', color='k', markersize=1, linestyle = 'None', label='Train Snapshots'),
                    Line2D([0], [0], color='black', lw=1, label='Boundaries'),
                    Line2D([0], [0], marker='x', color='red', linestyle = 'None', label='Centers'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='magenta', markersize=5, label='New Snapshot')]

axs[0].set_title("$k$ too small\nFail to detect novelty")
axs[1].set_title("$k$ right")
axs[2].set_title("$k$ too large")


# Add legend with custom elements
fig.legend(handles=legend_elements, loc='outside lower center', ncol=4)
    
plt.show()
