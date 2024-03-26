# %%
# generate the model with noise reduciton
from matplotlib import projections, use
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
from matplotlib import cm
import src
import importlib
from sklearn.metrics import silhouette_score, silhouette_samples
from rich import print
import pandas as pd
import pickle
from os import path
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.vis.set_matplotlib_params()

# script settings
featfilepath = r"data\processed\ETEL_Test1\train_data.csv"   # folder path
python_model_path = r"models\NormalVsNoisereduction"                          # python model file to be created and included in python code
features = pd.read_csv(featfilepath,sep='\t')
features = features.drop(columns=["Timestamp"]).dropna(axis=1)
pickleModelName = "ScaledModel_select.pickle"

define_importance_manually = False # if True, the importance of the features is defined manually
use_selectKbest = True # if True, the importance of the features is defined by SelectKBest, otherwise by Random Forest

# define importance manually
if define_importance_manually:
    feat_importance = np.concatenate((np.ones(7),np.zeros(60)))

# %% load the data
X = features.to_numpy() # data to fit in the model
print(np.shape(X))

# %% STANDARDIZE DATA
# mean - for all the features culumns
means = [np.mean(X[:,i]) for i in range(X.shape[1])]
# standard deviation - for all the features culumns
stds = [np.std(X[:,i]) for i in range(X.shape[1])]
# standardize the feature matrix
standardized_features_matrix = np.zeros(X.shape)
for i in range(X.shape[1]):
    standardized_features_matrix[:,i] = (X[:,i] - means[i])/stds[i] # remove the mean and divide by the standard deviation

# check if data is standardized correctly
if np.all(np.round([np.mean(standardized_features_matrix[:,i]) for i in range(X.shape[1])], 10) == 0) and np.all(np.round([np.std(standardized_features_matrix[:,i]) for i in range(X.shape[1])], 10) == 1):
    print("Data standardized successfully.")
else:
    raise Exception("Data not standardized correctly.")

# train a kmeans with different cluster number
sil_score=[]
range_clusters = range(2,10)
for n_clust in range_clusters:
    kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=10).fit(standardized_features_matrix)
    sil_score.append(silhouette_score(standardized_features_matrix, kmeans.labels_))
    print("n_clust:",n_clust,"silhouette_score:",silhouette_score(standardized_features_matrix, kmeans.labels_))

# %% plot the silhouette score
fig, ax = plt.subplots()
ax.plot(range_clusters,sil_score,c='k',marker='.')

# %% select the best number of clusters
n_clust = range_clusters[np.argmax(sil_score)] # selecting the max number of clusters
kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=10).fit(standardized_features_matrix)
print("kmeans trained with n_clust:",n_clust)

# %% fit model 
RF = RandomForestClassifier(random_state=0,verbose=True)
RF.fit(standardized_features_matrix,kmeans.labels_)

# verify accuracy of classificatoin
correct_predictions = RF.predict(standardized_features_matrix) == kmeans.labels_
print("Accuracy:",np.sum(correct_predictions)/len(correct_predictions))

# %% feature importance normalized
feat_importance = RF.feature_importances_/np.max(RF.feature_importances_)

# try with selectkbest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
selector = SelectKBest(k=10)
selector.fit(standardized_features_matrix, kmeans.labels_)
feat_importance_Kbest = selector.scores_/np.max(selector.scores_)

# %% select important features
fig, ax = plt.subplots()
x = np.array(range(len(feat_importance)))+1
ax.bar(x,feat_importance,color='#002B49', width=0.4, label='Random Forest')
ax.bar(x+0.4,feat_importance_Kbest,color='#EF7B00', width=0.4, label='SelectKBest')
ax.set_xlabel('feature')
ax.set_ylabel('weight')
ax.set_xticks(x+0.2)
ax.set_xticklabels(x,rotation=90)
ax.legend()


if use_selectKbest:
    feat_importance = feat_importance_Kbest

# retrain the model with the scaled features
# scale the features by importance
standardized_features_matrix = standardized_features_matrix*feat_importance
kmeans = KMeans(n_clusters=n_clust, random_state=0, n_init=10).fit(standardized_features_matrix)
print("kmeans trained with scaled features and n_clust:",n_clust)


# %% calculate radius of clusters
cluster_labels=kmeans.fit_predict(standardized_features_matrix) # y contains the assignments of each sample to a cluster
cluster_distances=kmeans.transform(standardized_features_matrix)
radiuses=[] # maximum distance to eah cluster in the train dataset
for cluster in range(0,n_clust): # for every cluster calculate the radius
    radiuses.append(max(cluster_distances[cluster_labels==cluster,cluster])) # y contains the assignments of each sample to a cluster
                                                                             # y_pred_train==cluster to select only the samples of this cluster
                                                                             # cluster_distances[:,cluster] to select the distances to this cluster
                                                                             # max() to select the maximum distance

# save the data to a file
np.savetxt("feature_importance.csv",feat_importance,delimiter=',')
# %% save the model to a python file
kmeans.radiuses = radiuses # add radiuses to the model
kmeans.feat_importance = feat_importance # add feature importance to the model
kmeans.means = means # add means to the model
kmeans.stds = stds # add standard deviation to the model
pickle.dump(kmeans, open(path.join(python_model_path,pickleModelName), 'wb'))

plt.show()