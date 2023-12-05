# %% TAKE DATA FROM UART, STANDARDIZE IT AN D PERFORM CLUSTERING
# %% IMPORTS
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

# %% global variables
timestamps = np.array([])
features_matrix = np.array([])
snapshots_filepath = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\putty.log"  # Change this to your actual file path
model_filepath = r"C:\Users\ariel\Documents\Courses\Tesi\Code\RTOS\Framework\Core\Inc\model.h"  # Change this to your actual file path
min_cluster_size = 2   # minimum number of samples in a cluster

# %% LOAD DATA
with open(snapshots_filepath, 'r') as file:
    for line in file:
        data = line.strip()
        if "Timestamp:" in data:
            timestamp_value = np.array([data.split("Timestamp: ")[1].rstrip('\n')])
            timestamps=np.append(timestamps,timestamp_value,axis=0)
        elif data == "Features:":
            features_line        = file.readline().strip()
            while features_line != "End of features.":
                features_values  = np.array([float(value) for value in features_line.split('\t')]).reshape(-1,1)
                if features_matrix.size == 0:
                    features_matrix = features_values.reshape(-1,1)
                else:
                    features_matrix  = np.concatenate((features_matrix,features_values),axis=1)
                features_line    = file.readline().strip()
features_matrix = np.transpose(features_matrix)
print("Data loaded successfully.")
# print("Timestamps: ", timestamps)       # timestamps.shape = (n_samples,)
# print("Features: ", features_matrix)    # features_matrix.shape = (n_samples, n_features)
print("Number of records: ", features_matrix.shape[0])
n_features = features_matrix.shape[1]
print("Number of features: ", n_features)

## %% STANDARDIZE DATA
# mean
means = [np.mean(features_matrix[:,i]) for i in range(features_matrix.shape[1])]
#print("Means: ", means)
# standard deviation
stds = [np.std(features_matrix[:,i]) for i in range(features_matrix.shape[1])]
#print("Standard deviations: ", stds)
# standardize
standardized_features_matrix = np.zeros(features_matrix.shape)
for i in range(features_matrix.shape[1]):
    standardized_features_matrix[:,i] = (features_matrix[:,i] - means[i])/stds[i]
#print("Standardized features: ", standardized_features_matrix)

# check if data is standardized correctly
if np.all(np.round([np.mean(standardized_features_matrix[:,i]) for i in range(features_matrix.shape[1])], 10) == 0) and np.all(np.round([np.std(standardized_features_matrix[:,i]) for i in range(features_matrix.shape[1])], 10) == 1):
    print("Data standardized successfully.")
else:
    raise Exception("Data not standardized correctly.")

# %% PERFORM CLUSTERING
# train with different number of clusters
sil_score=[]
inertia=[]
max_clusters=min(100,features_matrix.shape[0]) # it's not possible to have more clusters than samples
for n_clusters in range(1,max_clusters+1):
    kmeans=KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=10)
    cluster_labels=kmeans.fit_predict(standardized_features_matrix)
    if 1<n_clusters<max_clusters:
        sil_score.append(silhouette_score(standardized_features_matrix,cluster_labels))
    inertia.append(kmeans.inertia_)

# PLOT SILHOUETTE SCORE and INERTIA
fig, axs=plt.subplots(2,1)

axs[0].plot(range(2,max_clusters),sil_score)
axs[0].set_ylabel('Silhouette')

axs[1].plot(range(1,max_clusters+1),inertia)
axs[1].set_ylabel('Inertia')
axs[1].sharex(axs[0])
axs[0].set_xlabel('Num. of clusters')

fig.tight_layout()
plt.show()

# %% select number of clusters and plot results

# select clusters according to silhouette analisys - train dataset
n_clusters = input("Enter desired number of clusters: ")
n_clusters = int(n_clusters)

kmeans=KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=10)
cluster_labels=kmeans.fit_predict(standardized_features_matrix) # y contains the assignments of each sample to a cluster

range_feature_1=range(30, 33) # select features to plot vs next ones
range_feature_2=range(50, 53) # select features to plot vs previous ones

fig, axs=plt.subplots(len(range_feature_1),len(range_feature_2),sharex=True,sharey=True)
fig.tight_layout()

for i in range_feature_1: 
    for j in range_feature_2:
        if i!=j:
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(standardized_features_matrix[:,i],standardized_features_matrix[:,j],s=1,marker='*',c=cluster_labels+1,cmap='Set1')
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x',c=range(1,n_clusters+1),cmap='Set1')
        if i==range_feature_1[-1]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_xlabel('feature '+str(j))
        if j==range_feature_2[0]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_ylabel('feature '+str(i))

plt.show()

# %% force small clusters to merge with the closest cluster
# Check the size of each cluster
# cluster_sizes = np.bincount(cluster_labels)

# # Identify clusters with fewer elements than the specified minimum
# small_clusters = np.where(cluster_sizes < min_cluster_size)[0]

# # Iterate over small clusters and reassign points to the nearest larger cluster
# for small_cluster in small_clusters:
#     # Find the indices of points in the small cluster
#     indices = np.where(cluster_labels == small_cluster)[0]

#     # Calculate pairwise distances to the centroids of other clusters
#     distances = pairwise_distances_argmin_min(standardized_features_matrix[indices,:], kmeans.cluster_centers_)[0]

#     # Find the nearest cluster with a sufficient number of elements
#     nearest_large_cluster = np.argmin(cluster_sizes[cluster_sizes >= min_cluster_size])

#     # Reassign points to the nearest large cluster
#     cluster_labels[indices] = nearest_large_cluster

# %% calculate radius of clusters
cluster_distances=kmeans.transform(standardized_features_matrix)
radiuses=[] # maximum distance to eah cluster in the train dataset
for cluster in range(0,n_clusters): # for every cluster calculate the radius
    radiuses.append(max(cluster_distances[cluster_labels==cluster,cluster])) # y contains the assignments of each sample to a cluster
                                                                            # y_pred_train==cluster to select only the samples of this cluster
                                                                            # cluster_distances[:,cluster] to select the distances to this cluster
                                                                            # max() to select the maximum distance

# %% print results for implementation in C code to model file
with open(model_filepath, 'w') as f:
    # n_clusters
    f.write("int n_clusters = "+str(n_clusters)+";\n")

    # centers
    aux = "{ {"
    for i in range(n_clusters):
        for j in range(features_matrix.shape[1]):
            aux += str(kmeans.cluster_centers_[i][j])
            if j != features_matrix.shape[1]-1:
                aux += ", "
        if i != n_clusters-1:
            aux += "}, {"
        else:
            aux += "} };"
    f.write("double centers["+str(n_clusters)+"]["+str(features_matrix.shape[1])+"] = "+aux+"\n")

    # stds
    aux = ""
    for i in range(n_features):
        aux += str(stds[i])
        if i != n_features-1:
            aux += ", "
    f.write("double stds["+str(n_features)+"] = {"+aux+"};\n")

    # means
    aux = ""
    for i in range(n_features):
        aux += str(means[i])
        if i != n_features-1:
            aux += ", "
    f.write("double means["+str(n_features)+"] = {"+aux+"};\n")

    # clusters radius
    aux = ""
    for i in range(n_clusters):
        aux += str(radiuses[i])
        if i != n_clusters-1:
            aux += ", "
    f.write("double radiuses["+str(n_clusters)+"] = {"+aux+"};\n")
print("Model file created successfully.")
f.close()   # close file
