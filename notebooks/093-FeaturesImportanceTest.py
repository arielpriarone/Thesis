# %%
# this don't work well because the cluster have differnt sizes
import dis
from matplotlib import projections
from k_means_constrained import KMeansConstrained as KMeans
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
featfilepath = r"data\processed\ETEL_Test2\train_data.csv"              # folder path - test now with the ETEL_Test2 data
faultfilepath = r"data\processed\ETEL_Test2\withfault\test_data.csv"   # folder path - test now with the ETEL_Test2 data
python_model_path = r"models\NormalVsNoisereduction"                    # python model file to be created and included in python code

for filepath in [featfilepath, faultfilepath]:
    features = pd.read_csv(filepath,sep='\t')
    try:
        features = features.drop(columns=["Timestamp"]).dropna(axis=1)
    except:
        print("Timestamp column not found")
    try:
        features = features.drop(["Metric"])
    except:
        print("Metric column not found")
    print(features.keys())
    print(features.head())

    # %% load the data
    X = features.to_numpy() # data to fit in the model
    print(np.shape(X))

    # load the models
    model = pickle.load(open(path.join(python_model_path,"StandardModel.pickle"), 'rb'))
    scaledModel = pickle.load(open(path.join(python_model_path,"ScaledModel.pickle"), 'rb'))

    # print the models
    print(f"modelradiuses = {model.radiuses}")
    print(f"scaledModelradiuses = {scaledModel.radiuses}")

    # %% STANDARDIZE DATA
    # mean - for all the features culumns
    standardized_features_matrix = np.array([(x-model.means)/model.stds for x in X])
    print(standardized_features_matrix.shape)
    scaled_features_matrix = np.array([(x-scaledModel.means)/scaledModel.stds*scaledModel.feat_importance for x in X])

    metric = {"standard": [], "scaled": []}
    # evaluate the models on the training data collected the second day
    for i in range(standardized_features_matrix.shape[0]): # standard model
        y = model.predict(standardized_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = model.transform(standardized_features_matrix[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-model.radiuses[int(y)])/model.radiuses[int(y)]) # calculate the error
        metric['standard'].append(current_error)
    for i in range(scaled_features_matrix.shape[0]): # scaled model
        y = model.predict(scaled_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = model.transform(scaled_features_matrix[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-model.radiuses[int(y)])/model.radiuses[int(y)]) # calculate the error
        metric['scaled'].append(current_error)

    fig, ax = plt.subplots()
    ax.plot(metric['standard'], label='Standard novelty metric')
    ax.plot(metric['scaled'], label='Scaled novelty metric')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Novelty metric')
    ax.legend()

plt.show()
