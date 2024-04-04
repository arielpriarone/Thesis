import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

from rich import print
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# generate two blobs
X, y = make_blobs(n_samples=400, centers=2, n_features=2, random_state=1)

# plot the blobls
fig, ax = plt.subplots()
# generate a colormap with two colors defined as hex
cmap = plt.cm.colors.ListedColormap(['#002B49', '#EF7B00'])
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, marker='.', s=10)

C1=X[y==0]
C2=X[y==1]

Cent1=np.mean(C1,axis=0)
Cent2=np.mean(C2,axis=0)

ax.set_aspect('equal')

# calculate the radii
Radius1 = np.linalg.norm(C1 - Cent1, axis=1).max()
Radius2 = np.linalg.norm(C2 - Cent2, axis=1).max()

# plot the circles
circle1 = plt.Circle(Cent1, Radius1, color='#002B49', fill=False)
circle2 = plt.Circle(Cent2, Radius2, color='#EF7B00', fill=False)

ax.add_artist(circle1)
ax.add_artist(circle2)

new_instance = np.array([-4, 10])
ax.scatter(new_instance[0], new_instance[1], c='magenta', marker='.', s=100)

# plot the arrow
ax.quiver(Cent1[0], Cent1[1], new_instance[0]-Cent1[0], new_instance[1]-Cent1[1], angles='xy', scale_units='xy', scale=1, color='#002B49')
ax.quiver(Cent2[0], Cent2[1], new_instance[0]-Cent2[0], new_instance[1]-Cent2[1], angles='xy', scale_units='xy', scale=1, color='#EF7B00')


plt.show()