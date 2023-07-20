# %%
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
import matplotlib
import src
import importlib
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'

n_features=4
n_blobs=8
X,y=make_blobs(1000,n_features,centers=n_blobs, random_state=655)
print(X[0,:])
print(y)

# %%
kmeans=KMeans(n_blobs)
y_pred=kmeans.fit_predict(X)

fig, axs=plt.subplots(n_features,n_features,sharex=True,sharey=True)
fig.tight_layout()
for i in range(0, n_features): 
    for j in range(0, n_features):
        if i!=j:
            axs[i,j].scatter(X[:,i],X[:,j],c=y,cmap='tab20b',s=1,marker='.')
            axs[i,j].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x')
plt.show()

# %%
fig, axs=plt.subplots(n_features,n_features,sharex=True,sharey=True)
fig.tight_layout()
for i in range(0, n_features): 
    for j in range(0, n_features):
        if i!=j:
            axs[i,j].scatter(X[:,i],X[:,j],c=y_pred,cmap='tab20b',s=1,marker='.')
            axs[i,j].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x')
plt.show()



# %%

