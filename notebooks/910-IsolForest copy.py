from cProfile import label
from tkinter import font
from matplotlib import legend
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import src
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from rich import print
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
# %% fit model 
src.vis.set_matplotlib_params()
matplotlib.rcParams['figure.figsize'] = (matplotlib.rcParams['figure.figsize'][0], matplotlib.rcParams['figure.figsize'][1]*0.8) # set size of plots

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

# %% fit model
clf = IsolationForest(random_state=0,verbose=True,max_samples=np.shape(X)[0])
clf.fit(X_aniso)

# %% plot the densities
x, y = np.meshgrid(np.linspace(-4,4,100),np.linspace(-5,5,100))
Z = np.exp(clf.decision_function(np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)))

print("Shape:",Z.shape)
print("Max dec. funct:",np.max(Z))
print("Min dec. funct:",np.min(Z))

fig, ax = plt.subplots()
plot = ax.contourf(x,y,Z.reshape(x.shape),levels=100,cmap='viridis',label='Density')
ax.scatter(X_aniso[:,0],X_aniso[:,1],s=1,marker='.',c='black',label='Data points')
ax.set_xlim([-4,4])
ax.set_ylim([-5,5])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.set_aspect
cbar=fig.colorbar(plot)
cbar.ax.set_ylabel('Decision function')
fig.tight_layout()

plt.show()
# %%
