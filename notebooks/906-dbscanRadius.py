from tkinter import font
from matplotlib import legend
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import src
from matplotlib.lines import Line2D

src.vis.set_matplotlib_params()

# Generate random points
points = datasets.make_blobs(n_samples=60, centers=1, random_state=1)[0]

# Choose a center point
center = points[18]

# Set the radius of the circle
epsilon = 0.5

# Plot the points
plt.scatter(points[:, 0], points[:, 1], color='black', marker='.')



# Plot the points inside the circle in purple
distances = np.linalg.norm(points - center, axis=1)
inside_circle = distances <= epsilon
plt.scatter(points[inside_circle][:, 0], points[inside_circle][:, 1], color='m', marker='.')

# Draw the circle
circle = plt.Circle(center, epsilon, color='k', fill=False)
plt.gca().add_patch(circle)

# Plot the center point in red
plt.scatter(center[0], center[1], color='red', marker='.')

# Set the aspect ratio to equal
plt.axis('equal')

# Choose a center point
center = points[19]
# Plot the points inside the circle in purple
distances = np.linalg.norm(points - center, axis=1)
inside_circle = distances <= epsilon
plt.scatter(points[inside_circle][:, 0], points[inside_circle][:, 1], color='m', marker='.')

# Draw the circle
circle = plt.Circle(center, epsilon, color='k', fill=False)
plt.gca().add_patch(circle)

# Plot the center point in red
plt.scatter(center[0], center[1], color='red', marker='.')

# text
plt.text(-1.6,4.1, "$p_1$", fontsize=12, color='k', ha='center', va='center')
plt.text(-2.4,5.45, "$p_2$", fontsize=12, color='k', ha='center', va='center')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

legend_elements = [ Line2D([0], [0], marker='.',markersize = 8, color='w',markerfacecolor="r", label='considered points'),
                    Line2D([0], [0], marker='.',markersize = 8, color='w',markerfacecolor='m', label=r'points within $\varepsilon$ to considered points'),
                    Line2D([0], [0], marker='.',markersize = 8, color='w',markerfacecolor='k', label='other points')]
plt.legend(handles=legend_elements, loc='upper left',bbox_to_anchor=(1, 1),fontsize=10)
plt.tight_layout()
# Show the plot
plt.show()
