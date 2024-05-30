from os import path
from rich import print
from rich.console import Console
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import ceil, floor
from sklearn.metrics import silhouette_score
import datetime
import matplotlib as mpl
import typer
import pickle

# %% init
console = Console()

# %% global variables
timestamps = np.array([])                                                                       # timestamps.shape = (n_samples,)
features_matrix = np.array([])                                                                  # features_matrix.shape = (n_samples, n_features)
train_data_filepath = r"data\processed\ETEL_Test2\train_data_refined_subset7features.csv"    # csv file with train data
feature_scaler_filepath = r".\feature_importance.csv"    # csv file with feature scaler
model_filepath = r"RTOS\Restored\Core\Inc\model.h"                                              # model file to be created and included in C.
modelfilename = "Model_7_feat_20240227.pickle"                                                          # model file to be created and included in python code
python_model_path = r"models\NormalVsNoisereduction"                          # python model file to be created and included in python code
max_n_clusters = 25                                                                             # maximum number of clusters to try
min_cluster_size = 2                                                                            # minimum number of samples in a cluster
use_weights = False                                                                             # use feature weights

# %% LOAD DATA
console.print("Loading data...", style="magenta")
train_data = pd.read_csv(train_data_filepath, sep='\t', header=1)
console.print(train_data.head(), style="magenta")
timestamps = train_data.iloc[:,0].to_numpy()
features_matrix = train_data.iloc[:,1:].to_numpy()
feat_weights = np.ones(features_matrix.shape[1])
if use_weights:
    try:
        feat_weights = np.array(pd.read_csv(feature_scaler_filepath)).flatten()
    except:
        console.print("Feature importance file not found, all features will be considered with equal weights.", style="magenta")
        if not typer.confirm("do you want to proceed?", abort=True):
            exit()
    
console.print("Data loaded successfully.", style="magenta")
console.print("Number of records: ", features_matrix.shape[0], style="magenta")
n_features = features_matrix.shape[1] # 67 - number of features
console.print("Number of features: ", n_features,style="magenta")

# %% STANDARDIZE DATA
# mean - for all the features culumns
means = [np.mean(features_matrix[:,i]) for i in range(features_matrix.shape[1])]
# standard deviation - for all the features culumns
stds = [np.std(features_matrix[:,i]) for i in range(features_matrix.shape[1])]
# standardize the feature matrix
standardized_features_matrix = np.zeros(features_matrix.shape)
for i in range(features_matrix.shape[1]):
    standardized_features_matrix[:,i] = (features_matrix[:,i] - means[i])/stds[i] # remove the mean and divide by the standard deviation

# check if data is standardized correctly
if np.all(np.round([np.mean(standardized_features_matrix[:,i]) for i in range(features_matrix.shape[1])], 10) == 0) and np.all(np.round([np.std(standardized_features_matrix[:,i]) for i in range(features_matrix.shape[1])], 10) == 1):
    print("Data standardized successfully.")
else:
    raise Exception("Data not standardized correctly.")

# %% PERFORM CLUSTERING
# train with different number of clusters
sil_score=[]    # silhouette score
inertia=[]      # inertia
max_clusters=min(max_n_clusters,floor(features_matrix.shape[0]/min_cluster_size))   # it's not possible to have more clusters than samples/min_cluster_size
for n_clusters in range(1,max_clusters+1):
    kmeans=KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=10)#, size_min=min_cluster_size, size_max=features_matrix.shape[0])
    cluster_labels=kmeans.fit_predict(standardized_features_matrix)
    if 1<n_clusters<max_clusters:
        sil_score.append(silhouette_score(standardized_features_matrix,cluster_labels))
    inertia.append(kmeans.inertia_)
    console.print(f"Kmeans with {n_clusters} clusters completed.", style="magenta")

# PLOT SILHOUETTE SCORE and INERTIA
fig, axs=plt.subplots(2,1)

axs[0].plot(range(2,max_clusters),sil_score)
axs[0].set_ylabel('Silhouette')

axs[1].plot(range(1,max_clusters+1),inertia)
axs[1].set_ylabel('Inertia')
axs[1].sharex(axs[0])
axs[0].set_xlabel('Num. of clusters')

fig.tight_layout()

console.print("Decide the number of clusters based on the plot, and close the plot window.", style="magenta")
plt.show()

# %% select number of clusters and plot results

# select clusters according to silhouette analisys - train dataset
n_clusters = input("Enter desired number of clusters: ")
n_clusters = int(n_clusters)

kmeans=KMeans(n_clusters=n_clusters, init='k-means++', max_iter=1000, n_init=10)
cluster_labels=kmeans.fit_predict(standardized_features_matrix) # y contains the assignments of each sample to a cluster

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
    # header
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"/* MODEL CREATED AUTOMATICALLY BY notebooks/203-MicroGenerateModel.py, timestamp: {timestamp} */\n\n")
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

    # feature importance scaler
    aux = ""
    for i in range(n_features):
        aux += str(feat_weights[i])
        if i != n_features-1:
            aux += ", "
    f.write("double weights["+str(n_features)+"] = {"+aux+"};\n")

console.print("Model file created successfully.", style="magenta")
f.close()   # close file

# %% plot resulting clusters
try:
    range_feature_1=range(30, 33) # select features to plot vs next ones
    range_feature_2=range(50, 53) # select features to plot vs previous ones

    fig, axs=plt.subplots(len(range_feature_1),len(range_feature_2),sharex=True,sharey=True)
    fig.tight_layout()
    cmap = mpl.colormaps['Set1']
    for i in range_feature_1: 
        for j in range_feature_2:
            if i!=j:
                axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(standardized_features_matrix[:,i],standardized_features_matrix[:,j],s=1,marker='*',c=cluster_labels+1,cmap='Set1')
                axs[i-range_feature_1[0],j-range_feature_2[0]].scatter(kmeans.cluster_centers_[:,i],kmeans.cluster_centers_[:,j],marker='x',c=range(1,n_clusters+1),cmap='Set1')
            if i==range_feature_1[-1]:
                axs[i-range_feature_1[0],j-range_feature_2[0]].set_xlabel('feature '+str(j))
            if j==range_feature_2[0]:
                axs[i-range_feature_1[0],j-range_feature_2[0]].set_ylabel('feature '+str(i))
    #        for cluster in range(0,n_clusters):
    #            axs[i-range_feature_1[0],j-range_feature_2[0]].add_patch(plt.Circle((kmeans.cluster_centers_[cluster,i],kmeans.cluster_centers_[cluster,j]),radiuses[cluster],color=cmap(cluster),fill=False))
            axs[i-range_feature_1[0],j-range_feature_2[0]].set_aspect('equal') # set aspect ratio to 1 to preserve circle shape
    fig.tight_layout()
except:
    console.print("Error plotting clusters.", style="magenta")

# %% save the model to a python file
kmeans.radiuses = radiuses # add radiuses to the model
kmeans.means = means
kmeans.stds = stds
kmeans.feat_weights = feat_weights
pickle.dump(kmeans, open(path.join(python_model_path,modelfilename), 'wb'))


plt.show()
