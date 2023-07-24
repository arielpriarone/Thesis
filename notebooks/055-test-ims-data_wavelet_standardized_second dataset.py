# to do: fix this script so it stores conveniently the second dataset, and then repeat the feature extraction

# %%
import matplotlib
matplotlib.use('Qt5Agg')
import importlib
import pywt
import matplotlib.pyplot as plt
import numpy as np
import os
import src
import pickle 
import tikzplotlib as mp2tk
from sklearn.preprocessing import StandardScaler
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
auxpath=''                                                  # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
    
# script settings
dirPath     = auxpath + "./data/raw/2nd_test_IMSBearing/"   # folder path
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized_second.pickle") #file to save the analisys
tickpath    = os.path.join(auxpath + "./reports/tickz/")    #file to save the tickz

decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
TrainingData={}                                             # empty dictionary to save data 
n_split     = 1500                                          # number of sample to split the dataset

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


stdsclr=StandardScaler()
indx=0
fileList=[]
if decompose:
    for fileName in os.listdir(dirPath):
        # check if current path is a file
        if os.path.isfile(os.path.join(dirPath, fileName)):# and indx<6:
            fileList.append(fileName)
            snap=src.data.snapshot()
            snap.readImsFile(path=dirPath+fileName, variables="Bearing 3 x")

            # Frequency domain analisys
            samplFreq=20000 #hz
            y=snap.rawData["Bearing 3 x"].to_numpy()
            #FFT, FFTfrequencies, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc="Hann")
            # wavelet packet
            wp = pywt.WaveletPacket(data=snap.rawData["Bearing 3 x"], wavelet='db10', mode='symmetric',maxlevel=6)
            nodes=[node.path for node in wp.get_level(wp.maxlevel, 'natural')]
            powers=[np.linalg.norm(wp[index].data) for index in nodes]  # getting the powers of the wavelet coefs
            if indx==0:
                wavanaly=powers
            else:
                wavanaly = np.vstack([wavanaly, powers])
            if indx==n_split: #split training dataset and validation dataset
                stdsclr.fit(wavanaly) 
            indx+=1
            print(indx)
    wavanaly_standardized=stdsclr.transform(wavanaly)

    TrainingData['wavanaly']=wavanaly
    TrainingData['wavanaly_standardized']=wavanaly_standardized
    TrainingData['wavanaly_train']=wavanaly[0:n_split,:]
    TrainingData['wavanaly_standardized_train']=wavanaly_standardized[0:n_split,:]
    TrainingData['wavanaly_test']=wavanaly[n_split::,:]
    TrainingData['wavanaly_standardized_test']=wavanaly_standardized[n_split::,:]
    TrainingData['nodes']=nodes
    filehandler = open(savepath, 'wb') 
    pickle.dump(TrainingData, filehandler)
    filehandler.close()
else:
    filehandler = open(savepath, 'rb') 
    TrainingData = pickle.load(filehandler)

#%%
from mpl_toolkits import mplot3d
x = np.arange(0,TrainingData['wavanaly_standardized'].shape[1],1)
y = np.arange(0,TrainingData['wavanaly_standardized'].shape[0],1)
X,Y = np.meshgrid(x,y)
# Plot a 3D surface
print(X.shape)
print(Y.shape)
print(TrainingData['wavanaly_standardized'].shape)
fig = plt.figure('figure1')#,figsize=[15, 15])
ax = plt.axes(projection='3d')
ax.set_xlabel('Features')
ax.set_ylabel('Snapshots')
ax.set_zlabel('Amplitude')
#ax.yaxis.set_ticks(np.arange(0,6,1))
dummy=np.round(np.linspace(0, len(fileList) - 1, 6)).astype(int).tolist()
print(dummy)
#ax.set_yticklabels([fileList[i] for i in dummy])
# Create surface plot
ax.scatter(X, Y, TrainingData['wavanaly_standardized'], marker='.',c=TrainingData['wavanaly_standardized']/np.max(TrainingData['wavanaly_standardized']), cmap='turbo')
ax.set_xticklabels(TrainingData['nodes'])


# %%
# analyze just one snapshot
aux=200 # number of the considered sample

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(TrainingData['nodes'],TrainingData['wavanaly'][aux,:])
ax.tick_params(labelbottom=False)
ax.set_ylabel('Amplitude')
mp2tk.save(tickpath+'WT_SingSnap_a.tex',axis_height='3cm', axis_width='0.9\linewidth')

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(TrainingData['nodes'],TrainingData['wavanaly'][aux,:]/np.linalg.norm(TrainingData['wavanaly'][aux,:]))
ax.tick_params(labelbottom=False)
ax.set_ylabel('Amplitude')
mp2tk.save(tickpath+'WT_SingSnap_b.tex',axis_height='3cm', axis_width='294.76926pt')

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(TrainingData['nodes'],TrainingData['wavanaly_standardized'][aux,:])
ax.set_ylabel('Amplitude')
ax.tick_params(axis='x',rotation=90)
mp2tk.save(tickpath+'WT_SingSnap_c.tex',axis_height='3cm', axis_width='294.76926pt')




#%%
# print as a heatmap
mynorm = plt.Normalize(vmin=np.min(TrainingData['wavanaly_standardized']), vmax=np.max(TrainingData['wavanaly_standardized']))
sm = plt.cm.ScalarMappable(cmap='plasma', norm=mynorm)
fig, axs = plt.subplots(1,2)
fig.tight_layout()
aux=200 # number of the considered sample
im=axs[0].imshow(TrainingData['wavanaly_standardized'][0:0+2*65,:],cmap='plasma')
im.set_norm(mynorm)
axs[0].set_xticklabels(TrainingData['nodes'])
axs[0].set_xlabel('Features')
axs[0].set_ylabel('n° of record')
axs[0].set_title('Normal Functioning')

start=2028
im=axs[1].imshow(TrainingData['wavanaly_standardized'][start:start+2*65,:],cmap='plasma')
im.set_norm(mynorm)
axs[1].set_xticklabels(TrainingData['nodes'])
axs[1].set_yticklabels(np.arange(start,start+2*65))
axs[1].set_xlabel('Features')
axs[1].set_ylabel('n° of record')
axs[1].set_title('Abnormal Functioning')

cb=fig.colorbar(im)
cb.set_label('Power of the feature')
plt.show()
# %%
