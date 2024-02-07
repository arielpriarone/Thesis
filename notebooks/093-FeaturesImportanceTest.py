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
import matplotlib as mpl
matplotlib.use('Qt5Agg')
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.vis.set_matplotlib_params()
mpl.rcParams['lines.linewidth'] = 0.5

def mobileAverage(data, window=5):
    """Calculate the mobile average of the data"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

# script settings
featfilepath = r"data\processed\ETEL_Test2\train_data.csv"              # folder path - test now with the ETEL_Test2 data
faultfilepath = r"data\processed\ETEL_Test2\withfault\train_data.csv"   # folder path - test now with the ETEL_Test2 data
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
    standardModel = pickle.load(open(path.join(python_model_path,"StandardModel.pickle"), 'rb'))
    scaledModel = pickle.load(open(path.join(python_model_path,"ScaledModel.pickle"), 'rb'))
    scaledModel_subset = pickle.load(open(path.join(python_model_path,"ScaledModel_refined_subset.pickle"), 'rb'))
    standardModel_refined = pickle.load(open(path.join(python_model_path,"StandardModel_refined.pickle"), 'rb'))
    standardModel_notworking = pickle.load(open(path.join(python_model_path,"StandardModel_notworking.pickle"), 'rb'))

    # print the models
    print(f"modelradiuses = {standardModel.radiuses}")
    print(f"scaledModelradiuses = {scaledModel.radiuses}")

    # %% STANDARDIZE DATA
    # mean - for all the features culumns
    standardized_features_matrix = np.array([(x-standardModel.means)/standardModel.stds for x in X])
    standardized_features_matrix_refined = np.array([(x-standardModel_refined.means)/standardModel_refined.stds for x in X])
    standardized_features_matrix_notworking = np.array([(x-standardModel_notworking.means)/standardModel_notworking.stds for x in X])
    scaled_features_matrix = np.array([(x-scaledModel.means)/scaledModel.stds*scaledModel.feat_importance for x in X])
    scaled_features_matrix_subset = np.array([(x-scaledModel_subset.means)/scaledModel_subset.stds*scaledModel_subset.feat_importance for x in X])

    metric = {"standard": [], 
              "scaled": [], 
              "standard refined": [], 
              "scaled subset": [], 
              "standard notworking": []}
    # evaluate the models on the training data collected the second day
    for i in range(standardized_features_matrix.shape[0]): # standard model
        y = standardModel.predict(standardized_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel.transform(standardized_features_matrix[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel.radiuses[int(y)])/standardModel.radiuses[int(y)]) # calculate the error
        metric['standard'].append(current_error)
    for i in range(scaled_features_matrix.shape[0]): # scaled model
        y = scaledModel.predict(scaled_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = scaledModel.transform(scaled_features_matrix[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-scaledModel.radiuses[int(y)])/scaledModel.radiuses[int(y)]) # calculate the error
        metric['scaled'].append(current_error)
    for i in range(scaled_features_matrix_subset.shape[0]): # scaled model
        y = scaledModel_subset.predict(scaled_features_matrix_subset[i,:].reshape(1,-1))    
        distance_to_assigned_center = scaledModel_subset.transform(scaled_features_matrix_subset[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-scaledModel_subset.radiuses[int(y)])/scaledModel_subset.radiuses[int(y)]) # calculate the error
        metric['scaled subset'].append(current_error)
    for i in range(standardized_features_matrix_refined.shape[0]): # standard model
        y = standardModel_refined.predict(standardized_features_matrix_refined[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel_refined.transform(standardized_features_matrix_refined[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel_refined.radiuses[int(y)])/standardModel_refined.radiuses[int(y)]) # calculate the error
        metric['standard refined'].append(current_error)
    for i in range(standardized_features_matrix_notworking.shape[0]): # standard model
        y = standardModel_notworking.predict(standardized_features_matrix_notworking[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel_notworking.transform(standardized_features_matrix_notworking[i].reshape(1,-1))[0,y]
        print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel_notworking.radiuses[int(y)])/standardModel_notworking.radiuses[int(y)]) # calculate the error
        metric['standard notworking'].append(current_error)


    fig, ax = plt.subplots(2,1,sharex=True)
    fig.set_linewidth(0.5)
    ax[0].plot(metric['standard'], label='Standard novelty metric')
    ax[0].plot(metric['scaled'], label='Scaled novelty metric')
    ax[0].plot(metric['standard refined'], label='Standard refined novelty metric')
    ax[0].plot(metric['standard notworking'], label='Standard notworking novelty metric')
    ax[0].plot(metric['scaled subset'], label='Scaled subset novelty metric')
    ax[0].set_ylabel('Novelty metric')
    ax[0].legend()
    ax[0].set_title('Novelty metric comparison')
    ax[1].plot(mobileAverage(metric['standard']), label='Standard novelty metric')
    ax[1].plot(mobileAverage(metric['scaled']), label='Scaled novelty metric')
    ax[1].plot(mobileAverage(metric['standard refined']), label='Standard refined novelty metric')
    ax[1].plot(mobileAverage(metric['standard notworking']), label='Standard notworking novelty metric')
    ax[1].plot(mobileAverage(metric['scaled subset']), label='Scaled subset novelty metric')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Novelty metric')
    ax[1].set_title('Novelty metric comparison Moving Average')
    ax[1].legend()

plt.show()
