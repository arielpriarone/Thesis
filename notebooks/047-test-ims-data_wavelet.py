# %%
import matplotlib
matplotlib.use('Qt5Agg')
import importlib
import pywt
import matplotlib.pyplot as plt
import numpy as np
import os
import src
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath=''                                                  # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
src.visualization.set_matplotlib_params()                  # set nicer plot parameters
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
# script settings
dirPath     = auxpath + "./data/raw/1st_test_IMSBearing/"   # folder path
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
decompose   = False                                          # decompose using wavelet packet / reload previous decomposition
TrainingData={}                                             # empty dictionary to save data 
n_split     = 1500                                          # number of sample to split the dataset


indx=0
fileList=[]
for fileName in os.listdir(dirPath):
    # check if current path is a file
    if os.path.isfile(os.path.join(dirPath, fileName)):# and indx<6:
        fileList.append(fileName)
        snap=src.data.snapshot()
        snap.readImsFile(path=dirPath+fileName, variables="Bearing 3 x")
        # ax=snap.rawData.plot(x='time',y='Bearing 1 x',legend=False)
        # ax.grid(which='major',axis='both',color='grey', linestyle='-')
        # ax.grid(which='minor',axis='both',color='grey', linestyle=':')
        # ax.minorticks_on()
        # plt.figure(clear=True)

        # Frequency domain analisys
        samplFreq=20000 #hz
        y=snap.rawData["Bearing 3 x"].to_numpy()
        #FFT, FFTfrequencies, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc="Hann")
        # wavelet packet
        wp = pywt.WaveletPacket(data=snap.rawData["Bearing 3 x"], wavelet='db10', mode='symmetric',maxlevel=6)
        nodes=[node.path for node in wp.get_level(wp.maxlevel, 'natural')]
        powers=[np.linalg.norm(wp[index].data) for index in nodes]
        if indx==0:
            wavanaly=powers
        else:
            wavanaly = np.vstack([wavanaly, powers])       # Exclude sampling frequency
        indx+=1
        print(indx)
        # if indx==10: 
        #     break
#%%
from mpl_toolkits import mplot3d
x = np.arange(0,wavanaly.shape[1],1)
y = np.arange(0,wavanaly.shape[0],1)
X,Y = np.meshgrid(x,y)
# Plot a 3D surface
print(X.shape)
print(Y.shape)
print(wavanaly.shape)
fig = plt.figure('figure1')#,figsize=[15, 15])
ax = plt.axes(projection='3d')
# ax.set_xlabel('Features')
# ax.set_ylabel('Snapshots')
# ax.set_zlabel('amplitude')
#ax.yaxis.set_ticks(np.arange(0,6,1))
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
dummy=np.round(np.linspace(0, len(fileList) - 1, 6)).astype(int).tolist()
print(dummy)
#ax.set_yticklabels([fileList[i] for i in dummy])
# Create surface plot
ax.scatter(X, Y, wavanaly, marker='.',c=wavanaly/np.max(wavanaly), cmap='turbo')
plt.show()
# %%
