# %%
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
import numpy as np
import matplotlib
import src
import importlib
import pickle 
import os
from sklearn.metrics import silhouette_score, silhouette_samples
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'

# script settings
dirPath     = auxpath + "./data/raw/1st_test_IMSBearing/"   # folder path
savepath_data    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
savepath_model    = os.path.join(auxpath + "./models/", "kmeans_model.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                             # empty dictionary to save data 

filehandler = open(savepath_data, 'rb') 
IMSDATA = pickle.load(filehandler)


# %%
sil_score=[]
inertia=[]
max_clusters=15
for n_blobs in range(1,max_clusters+1):
    kmeans=KMeans(n_blobs)
    y_pred_train=kmeans.fit_predict(IMSDATA['wavanaly_standardized_train'])
    if n_blobs>1:
        sil_score.append(silhouette_score(IMSDATA['wavanaly_standardized_train'],y_pred_train))
    inertia.append(kmeans.inertia_)


# %%
fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(range(2,max_clusters+1),sil_score)
axs.set_ylabel('Silhouette')
axs.set_xlabel('Num. of clusters')

# %%
fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(range(1,max_clusters+1),inertia)
axs.set_ylabel('Inertia')
axs.set_xlabel('Num. of clusters')


# %%

# select 2 clusters because of silhouette analisys - train dataset
n_blobs=2
kmeans=KMeans(n_blobs)
y_pred_train=kmeans.fit_predict(IMSDATA['wavanaly_standardized_train'])

range_feature_1=range(30, 33)
range_feature_2=range(50, 53)

fig, axs=plt.subplots(len(range_feature_1),len(range_feature_2),sharex=True,sharey=True)
fig.tight_layout()

for i in range_feature_1: 
    for j in range_feature_2:
        print(i,j)
        if i!=j:
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(IMSDATA['wavanaly_standardized_train'][:,i],IMSDATA['wavanaly_standardized_train'][:,j],s=1,marker='.',c=y_pred_train+1,cmap='Set1')
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x',c='black')
        if i==range_feature_1[-1]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_xlabel('feature '+str(j))
        if j==range_feature_2[0]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_ylabel('feature '+str(i))


# %%
# with only train dataset - to check htat it's working ok
y_pred_train=kmeans.predict(IMSDATA['wavanaly_standardized_train'])
sil_samples=silhouette_samples(IMSDATA['wavanaly_standardized_train'],y_pred_train)

x0=0
xticks=[]
max_sil=[]
fig, axs=plt.subplots()
fig.tight_layout()
for i in range(n_blobs):
    y=sil_samples[y_pred_train==i]
    x=range(x0,x0+len(y))
    x0+=len(y)
    axs.fill_between(x,-np.sort(-y))
    xticks.append(np.mean(x))
    max_sil.append(max(y))
    axs.hlines(max_sil[i],x[0],x[-1],colors='red',linestyles='dashed')
axs.set_xticks(xticks)
axs.set_xticklabels(list(range(1,n_blobs+1)))
axs.set_ylabel('Silhouette')
axs.set_xlabel('Clusters')

# %%
# all dataset
y_pred_all=kmeans.predict(IMSDATA['wavanaly_standardized']) #only predict because all dataset


fig, axs=plt.subplots(len(range_feature_1),len(range_feature_2),sharex=True,sharey=True)
fig.tight_layout()

for i in range_feature_1: 
    for j in range_feature_2:
        print(i,j)
        if i!=j:
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(IMSDATA['wavanaly_standardized'][:,i],IMSDATA['wavanaly_standardized'][:,j],s=1,marker='.',c=y_pred_all+1,cmap='Set1')
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x',c='black')
        if i==range_feature_1[-1]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_xlabel('feature '+str(j))
        if j==range_feature_2[0]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_ylabel('feature '+str(i))


# %%
# with all dataset
y_pred_all=kmeans.predict(IMSDATA['wavanaly_standardized'])
sil_samples=silhouette_samples(IMSDATA['wavanaly_standardized'],y_pred_all)

x0=0
xticks=[]
fig, axs=plt.subplots()
fig.tight_layout()
for i in range(n_blobs):
    y=sil_samples[y_pred_all==i]
    x=range(x0,x0+len(y))
    x0+=len(y)
    axs.fill_between(x,-np.sort(-y))
    axs.hlines(max_sil[i],x[0],x[-1],colors='red',linestyles='dashed')
    xticks.append(np.mean(x))
axs.set_xticks(xticks)
axs.set_xticklabels(list(range(1,n_blobs+1)))
axs.set_ylabel('Silhouette')
axs.set_xlabel('Clusters')



# %%
# now i want to actually plot som ekind of error
# the idea is to record the max silhouette score and compare it with new sample
# the approach i want to try is to trasform the training dataset, and then, save the maximum distance found to each cluster, and then check 
# how far the new samples fall over the maximun distance recorded for the assigned cluster
cluster_distances=kmeans.transform(IMSDATA['wavanaly_standardized_train'])
max_dist=[] # maximum distance to eah cluster in the train dataset
for cluster in range(0,n_blobs):
    max_dist.append(max(cluster_distances[y_pred_train==cluster,cluster]))

# %% # apply this method first to just the train data to verify that error always < 0  
i=0; error=[]
for snap in IMSDATA['wavanaly_standardized_train']:
    # print(np.shape(snap))
    y=kmeans.predict(np.array(snap).reshape(1, -1)) # predict the cluster for the new snap
    error.append(kmeans.transform(np.array(snap).reshape(1, -1))[0,y]-max_dist[int(y)])
    
fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(error)
axs.set_xlabel('Samples')
axs.set_ylabel('error')

# %% # apply this method first to all data to verify that diverges
i=0; error=[]
for snap in IMSDATA['wavanaly_standardized']:
    # print(np.shape(snap))
    y=kmeans.predict(np.array(snap).reshape(1, -1)) # predict the cluster for the new snap
    error.append(kmeans.transform(np.array(snap).reshape(1, -1))[0,y]-max_dist[int(y)])
    
fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(error)
axs.hlines(0,0,len(error),colors=['k'],linestyles='dashdot')
axs.set_xlabel('Samples')
axs.set_ylabel('error')
axs.annotate('Novelty', xy = (1825, 1), 
             fontsize = 12, xytext = (1200, 100), 
             arrowprops = dict(facecolor = 'red'),
             color = 'k')

# %% # save the trained clustering
filehandler = open(savepath_model, 'wb') 
pickle.dump(kmeans, filehandler)
filehandler.close()


# %%
plt.show()