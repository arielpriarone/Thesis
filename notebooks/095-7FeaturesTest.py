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
import matplotlib.ticker as ticker
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
faultfilepath = r"data\processed\ETEL_Test2\withfault\test_data.csv"   # folder path - test now with the ETEL_Test2 data
python_model_path = r"models\NormalVsNoisereduction"                    # python model file to be created and included in python code

for filepath in [featfilepath,faultfilepath]:
    features = pd.read_csv(filepath,sep='\t')
    try:
        features = features.drop(columns=["Timestamp"]).dropna()
        print("Timestamp column dropped")
    except:
        print("Timestamp column not found")
    try:
        features = features.drop(columns=['Metric'])
        print("Metric column dropped")
    except:
        print("Metric column not found")
    features_7 = features.drop(columns=features.keys()[7:]) # take only the first 7 features
    print(features.keys())
    print(features.head())

    # %% load the data
    X = features.to_numpy() # data to fit in the model
    X_7 = features_7.to_numpy() # data to fit in the model with only first 7 features
    print(np.shape(X))

    # load the models
    standardModel = pickle.load(open(path.join(python_model_path,"StandardModel.pickle"), 'rb')) # this is the model trained with 1 day of data up to sample 399
    print("standardModel = ");print(standardModel)
    scaledModel = pickle.load(open(path.join(python_model_path,"ScaledModel.pickle"), 'rb')) # this is the model trained with 1 day of data up to sample 399
    print("scaledModel = ");print(scaledModel)
    scaledModel_subset = pickle.load(open(path.join(python_model_path,"ScaledModel_refined_subset.pickle"), 'rb'))  # this is the model trained with 1 day of data and partially the second day up to sample 499
    print("scaledModel_subset = ");print(scaledModel_subset)
    standardModel_refined = pickle.load(open(path.join(python_model_path,"StandardModel_refined.pickle"), 'rb')) #this is the standard model with all the second day of data in training dataset, except the noise
    print("standardModel_refined = ");print(standardModel_refined)
    standardModel_notworking = pickle.load(open(path.join(python_model_path,"StandardModel_notworking.pickle"), 'rb')) #this is the standard model with the noise in the dataset - not working properly
    print("standardModel_notworking = ");print(standardModel_notworking)
    standardModel_7features = pickle.load(open(path.join(python_model_path,"Standard_7features.pickle"), 'rb'))
    print("standardModel_7features = ");print(standardModel_7features)
    ScaledModel_7mask = pickle.load(open(path.join(python_model_path,"ScaledModel_7mask.pickle"), 'rb')) # this is the model with the mask to reduce the number of features - not working properly
    print("ScaledModel_7mask = ");print(ScaledModel_7mask)
    ScaledModel_Select = pickle.load(open(path.join(python_model_path,"ScaledModel_select.pickle"), 'rb')) # this is the model with the mask to reduce the number of features - not working properly
    print("ScaledModel_Select = ");print(ScaledModel_Select) # done with feature selectror

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
    standardized_features_matrix_7 = np.array([(x-standardModel_7features.means)/standardModel_7features.stds for x in X_7])
    standardized_features_matrix_7mask = np.array([(x-ScaledModel_7mask.means)/ScaledModel_7mask.stds for x in X])
    standardized_features_matrix_select = np.array([(x-ScaledModel_Select.means)/ScaledModel_Select.stds*ScaledModel_Select.feat_importance for x in X])
    
    metric = {"standard": [], 
              "scaled": [], 
              "standard refined": [], 
              "scaled subset": [], 
              "standard notworking": [],
              "7 features": [],
              "7 mask": [],
              "select": []}
    
    # evaluate the models on the training data collected the second day
    for i in range(standardized_features_matrix.shape[0]): # standard model
        y = standardModel.predict(standardized_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel.transform(standardized_features_matrix[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel.radiuses[int(y)])/standardModel.radiuses[int(y)]) # calculate the error
        metric['standard'].append(current_error)
    for i in range(scaled_features_matrix.shape[0]): # scaled model
        y = scaledModel.predict(scaled_features_matrix[i,:].reshape(1,-1))    
        distance_to_assigned_center = scaledModel.transform(scaled_features_matrix[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-scaledModel.radiuses[int(y)])/scaledModel.radiuses[int(y)]) # calculate the error
        metric['scaled'].append(current_error)
    for i in range(scaled_features_matrix_subset.shape[0]): # scaled model
        y = scaledModel_subset.predict(scaled_features_matrix_subset[i,:].reshape(1,-1))    
        distance_to_assigned_center = scaledModel_subset.transform(scaled_features_matrix_subset[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-scaledModel_subset.radiuses[int(y)])/scaledModel_subset.radiuses[int(y)]) # calculate the error
        metric['scaled subset'].append(current_error)
    for i in range(standardized_features_matrix_refined.shape[0]): # standard model
        y = standardModel_refined.predict(standardized_features_matrix_refined[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel_refined.transform(standardized_features_matrix_refined[i].reshape(1,-1))[0,y]
        # print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel_refined.radiuses[int(y)])/standardModel_refined.radiuses[int(y)]) # calculate the error
        metric['standard refined'].append(current_error)
    for i in range(standardized_features_matrix_notworking.shape[0]): # standard model
        y = standardModel_notworking.predict(standardized_features_matrix_notworking[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel_notworking.transform(standardized_features_matrix_notworking[i].reshape(1,-1))[0,y]
        # print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel_notworking.radiuses[int(y)])/standardModel_notworking.radiuses[int(y)]) # calculate the error
        metric['standard notworking'].append(current_error)
    for i in range(standardized_features_matrix_7.shape[0]): # standard model
        y = standardModel_7features.predict(standardized_features_matrix_7[i,:].reshape(1,-1))    
        distance_to_assigned_center = standardModel_7features.transform(standardized_features_matrix_7[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-standardModel_7features.radiuses[int(y)])/standardModel_7features.radiuses[int(y)]) # calculate the error
        metric['7 features'].append(current_error)
    for i in range(standardized_features_matrix_7mask.shape[0]): # standard model
        y = ScaledModel_7mask.predict(standardized_features_matrix_7mask[i,:].reshape(1,-1))    
        distance_to_assigned_center = ScaledModel_7mask.transform(standardized_features_matrix_7mask[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-ScaledModel_7mask.radiuses[int(y)])/ScaledModel_7mask.radiuses[int(y)]) # calculate the error
        metric['7 mask'].append(current_error)
    for i in range(standardized_features_matrix_select.shape[0]): # standard model
        y = ScaledModel_Select.predict(standardized_features_matrix_select[i,:].reshape(1,-1))    
        distance_to_assigned_center = ScaledModel_Select.transform(standardized_features_matrix_select[i].reshape(1,-1))[0,y]
        #print(f"y = {y}, distance = {distance_to_assigned_center}")
        current_error=float((distance_to_assigned_center-ScaledModel_Select.radiuses[int(y)])/ScaledModel_Select.radiuses[int(y)]) # calculate the error
        metric['select'].append(current_error)

    # exclude the 500 - 800 index range that are noise
    exclude = []
    if filepath == featfilepath:
        exclude = [i for i in range(500,800)] + [i for i in range(1,400)]  # noise set + first 400 samples
        
    metric['standard'] = [x for i,x in enumerate(metric['standard']) if not i in exclude]
    metric['scaled'] = [x for i,x in enumerate(metric['scaled']) if not i in exclude]
    metric['standard refined'] = [x for i,x in enumerate(metric['standard refined']) if not i in exclude]
    metric['scaled subset'] = [x for i,x in enumerate(metric['scaled subset']) if not i in exclude]
    metric['standard notworking'] = [x for i,x in enumerate(metric['standard notworking']) if not i in exclude]
    metric['7 features'] = [x for i,x in enumerate(metric['7 features']) if not i in exclude]
    metric['7 mask'] = [x for i,x in enumerate(metric['7 mask']) if not i in exclude]
    metric['select'] = [x for i,x in enumerate(metric['select']) if not i in exclude]

    fig, ax = plt.subplots(1,1)#, figsize=(7.68,5.78))
    fig.set_linewidth(0.5)
    ax.plot(metric['standard'], label='Model 1')              # label='Standard - train day 1')
    ax.plot(metric['scaled'], label='Model 2')    # label='Scaled Random Forest - train day 1')
    ax.plot(metric['select'], label='Model 3')            # label='Scaled scipy - train day 1')
    # ax.plot(metric['standard refined'], label='Standard all train novelty metric - train day 1 and 2')
    # ax.plot(metric['standard notworking'], label='Standard notworking novelty metric')
    ax.plot(metric['scaled subset'], label='Model 4')    # label='Scaled subset - train day 1 and partially 2')
    ax.plot(metric['7 features'], label='Model 5')                         # label='7 features novelty metric')
    #ax.plot(metric['7 mask'], label='7 mask novelty metric')

    ax.set_ylabel('Novelty metric [-]')

    # ax.set_title('Novelty metric comparison')
    # ax[1].plot(np.array(range(0,len(mobileAverage(metric['standard']))))+5,mobileAverage(metric['standard']))
    # ax[1].plot(np.array(range(0,len(mobileAverage(metric['standard']))))+5,mobileAverage(metric['scaled']))
    # ax[1].plot(np.array(range(0,len(mobileAverage(metric['standard']))))+5,mobileAverage(metric['select']))
    # # ax[1].plot(mobileAverage(metric['standard refined']))
    # # ax[1].plot(mobileAverage(metric['standard notworking']), label='Standard notworking novelty metric')
    # ax[1].plot(np.array(range(0,len(mobileAverage(metric['standard']))))+5,mobileAverage(metric['scaled subset']))
    # ax[1].plot(np.array(range(0,len(mobileAverage(metric['standard']))))+5,mobileAverage(metric['7 features']))
    #ax[1].plot(mobileAverage(metric['7 mask']), label='7 mask novelty metric')
   
    # ax[1].set_xlabel('Sample')
    # ax[1].set_ylabel('Novelty metric')
    # ax[1].set_title('Novelty metric comparison Moving Average (last 5 samples)')

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax[1].xaxis.set_major_locator(ticker.AutoLocator())
    # ax[1].xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(0, len(metric['standard']))
    # ax[1].set_xlim(0, len(metric['standard']))
    ax.hlines(0, 0, len(metric['standard']), color='grey', linestyle='-', label='__nolegend__')
    # ax[1].hlines(0, 0, len(metric['standard']), color='grey', linestyle='-', label='__nolegend__')
    ax.set_xlabel('Sample [-]')
    plt.subplots_adjust(
        top=0.845,
        bottom=0.215,
        left=0.083,
        right=0.985,
        hspace=0.2,
        wspace=0.2)
    fig.legend(loc='upper center', ncol = 5)

plt.show()
