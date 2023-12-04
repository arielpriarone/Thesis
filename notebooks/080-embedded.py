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

# %% global variables
timestamps = np.array([])
features_matrix = np.array([])
file_path = r"C:\Users\ariel\Documents\Courses\Tesi\Code\data\putty.log"  # Change this to your actual file path

# %% LOAD DATA
with open(file_path, 'r') as file:
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
for n_blobs in range(1,max_clusters+1):
    kmeans=KMeans(n_clusters=n_blobs, init='k-means++', max_iter=1000, n_init=10)
    y_pred_train=kmeans.fit_predict(standardized_features_matrix)
    if 1<n_blobs<max_clusters:
        sil_score.append(silhouette_score(standardized_features_matrix,y_pred_train))
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
n_blobs = 8 # n_blobs = input("Enter desired number of clusters: ")
n_blobs = int(n_blobs)

kmeans=KMeans(n_clusters=n_blobs, init='k-means++', max_iter=1000, n_init=10)
y_pred_train=kmeans.fit_predict(standardized_features_matrix)

range_feature_1=range(30, 33) # select features to plot vs next ones
range_feature_2=range(50, 53) # select features to plot vs previous ones

fig, axs=plt.subplots(len(range_feature_1),len(range_feature_2),sharex=True,sharey=True)
fig.tight_layout()

for i in range_feature_1: 
    for j in range_feature_2:
        if i!=j:
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(standardized_features_matrix[:,i],standardized_features_matrix[:,j],s=1,marker='*',c=y_pred_train+1,cmap='Set1')
            axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x',c=range(1,n_blobs+1),cmap='Set1')
        if i==range_feature_1[-1]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_xlabel('feature '+str(j))
        if j==range_feature_2[0]:
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_ylabel('feature '+str(i))

plt.show()

# %% print results for implementation in C code
aux = "{ {"
for i in range(n_blobs):
    for j in range(features_matrix.shape[1]):
        aux += str(kmeans.cluster_centers_[i][j])
        if j != features_matrix.shape[1]-1:
            aux += ", "
    if i != n_blobs-1:
        aux += "}, {"
    else:
        aux += "} };"
# centers
print("int centers["+str(n_blobs)+"]["+str(features_matrix.shape[1])+"] = "+aux)

# stds
aux = ""
for i in range(n_features):
    aux += str(stds[i])
    if i != n_features-1:
        aux += ", "
print("double stds["+str(n_features)+"] = {"+aux+"};")

# means
aux = ""
for i in range(n_features):
    aux += str(means[i])
    if i != n_features-1:
        aux += ", "
print("double means["+str(n_features)+"] = {"+aux+"};")
# %%
