# %%
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib.colors as color
import scipy as sp
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
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
TrainingData={}                                             # empty dictionary to save data 

filehandler = open(savepath, 'rb') 
TrainingData = pickle.load(filehandler)
sil_score=[]

# %%
for n_blobs in range(1,10):
    kmeans=KMeans(n_blobs)
    y_pred=kmeans.fit_predict(TrainingData['wavanaly_standardized'])
    sil_score[n_blobs]=silhouette_score(TrainingData['wavanaly_standardized'],y_pred)


# %%
fig, axs=plt.subplots()
fig.tight_layout()
axs.plot(n_blobs,sil_score)
plt.show()

