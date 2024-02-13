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

# %% gaussian mixture
#  training

max_clusters=15
BIC = [] # Bayesian Information Criterion
AIC = [] # Akaike Information Criterion
X = X_aniso # data to fit in the model
for n_blobs in range(1,max_clusters+1):
    GM = GaussianMixture(n_components=n_blobs, covariance_type='full', random_state=0)
    GM.fit(X)
    print(f'Number of clusters: {n_blobs}: the mixture model has converged: {GM.converged_}, with {GM.n_iter_} iterations')
    BIC.append(GM.bic(X))
    AIC.append(GM.aic(X))
# plot BIC and AIC
fig, ax = plt.subplots()
ax.plot(range(1,max_clusters+1),BIC,label='BIC', color='k', linestyle='--')
ax.plot(range(1,max_clusters+1),AIC,label='AIC', color='k', linestyle='dashdot')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Information Criterion')
ax.legend()
ax.annotate(r'Both criterion minimized for $k=3$ clusters', xy=(3, 900), xytext=(3,1200),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            )
fig.set_size_inches(np.multiply(fig.get_size_inches(),[1,0.7]))
fig.tight_layout()

# %% fit model with 3 clusters
GM = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
GM.fit(X)

# %% plot the densities
x, y = np.meshgrid(np.linspace(-4,4,100),np.linspace(-5,5,100))
Z = np.exp(GM.score_samples(np.concatenate((x.reshape(-1,1),y.reshape(-1,1)),axis=1)))

print("Shape:",Z.shape)
print("Max density:",np.max(Z))
print("Min density:",np.min(Z))

fig, ax = plt.subplots()
plot = ax.contourf(x,y,Z.reshape(x.shape),levels=100,cmap='viridis',label='Density')
ax.scatter(X[:,0],X[:,1],s=1,marker='.',c='black',label='Data points')
ax.scatter(GM.means_[:,0],GM.means_[:,1],s=100,marker='x',c='red',label='Cluster centers')
ax.set_xlim([-4,4])
ax.set_ylim([-5,5])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.set_aspect
cbar=fig.colorbar(plot)
cbar.ax.set_ylabel('Probability density')
fig.tight_layout()

# %% bell shape
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define mean and standard deviation
mu, sigma = 0, 1

# Generate x values
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# Calculate the PDF using the normal distribution
pdf = norm.pdf(x, mu, sigma)

# Plot the Gaussian distribution PDF
# Define mean and standard deviation
mu, sigma = 0.8, 0.3

# Generate x values
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)

# Calculate the PDF using the normal distribution
pdf = norm.pdf(x, mu, sigma)
fig, ax = plt.subplots()
ax.plot(x, pdf)
ax.set_xlabel('x')
ax.set_ylabel('Probability density')
ax.set_title(r'Normal Distribution PDF: $\mu=0.8$, $\sigma=0.3$')





plt.show()
# %%
