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
IMSDATA={}                                             # empty dictionary to save data 
n_split     = 600                                          # number of sample to split the dataset

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
            snap.readImsFile(path=dirPath+fileName, variables="Bearing 3",n_of_test=2)

            # Frequency domain analisys
            samplFreq=20000 #hz
            y=snap.rawData["Bearing 3"].to_numpy()
            #FFT, FFTfrequencies, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc="Hann")
            # wavelet packet
            wp = pywt.WaveletPacket(data=snap.rawData["Bearing 3"], wavelet='db10', mode='symmetric',maxlevel=6)
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

    IMSDATA['wavanaly']=wavanaly
    IMSDATA['wavanaly_standardized']=wavanaly_standardized
    IMSDATA['wavanaly_train']=wavanaly[0:n_split,:]
    IMSDATA['wavanaly_standardized_train']=wavanaly_standardized[0:n_split,:]
    IMSDATA['wavanaly_test']=wavanaly[n_split::,:]
    IMSDATA['wavanaly_standardized_test']=wavanaly_standardized[n_split::,:]
    IMSDATA['nodes']=nodes
    filehandler = open(savepath, 'wb') 
    pickle.dump(IMSDATA, filehandler)
    filehandler.close()
else:
    filehandler = open(savepath, 'rb') 
    IMSDATA = pickle.load(filehandler)

#%%
from mpl_toolkits import mplot3d
x = np.arange(0,IMSDATA['wavanaly_standardized'].shape[1],1)
y = np.arange(0,IMSDATA['wavanaly_standardized'].shape[0],1)
X,Y = np.meshgrid(x,y)
# Plot a 3D surface
print(X.shape)
print(Y.shape)
print(IMSDATA['wavanaly_standardized'].shape)
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
ax.scatter(X, Y, IMSDATA['wavanaly_standardized'], marker='.',c=IMSDATA['wavanaly_standardized']/np.max(IMSDATA['wavanaly_standardized']), cmap='turbo')
ax.set_xticklabels(IMSDATA['nodes'])


# %%
# analyze just one snapshot
aux=200 # number of the considered sample

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly'][aux,:])
ax.tick_params(labelbottom=False)
ax.set_ylabel('Amplitude')
mp2tk.save(tickpath+'WT_SingSnap_a.tex',axis_height='3cm', axis_width='0.9\linewidth')

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly'][aux,:]/np.linalg.norm(IMSDATA['wavanaly'][aux,:]))
ax.tick_params(labelbottom=False)
ax.set_ylabel('Amplitude')
mp2tk.save(tickpath+'WT_SingSnap_b.tex',axis_height='3cm', axis_width='294.76926pt')

fig, ax = plt.subplots(figsize=(6.4,1.5))
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly_standardized'][aux,:])
ax.set_ylabel('Amplitude')
ax.tick_params(axis='x',rotation=90)
mp2tk.save(tickpath+'WT_SingSnap_c.tex',axis_height='3cm', axis_width='294.76926pt')




#%%
# print as a heatmap
mynorm = plt.Normalize(vmin=np.min(IMSDATA['wavanaly_standardized']), vmax=np.max(IMSDATA['wavanaly_standardized']))
sm = plt.cm.ScalarMappable(cmap='plasma', norm=mynorm)
fig, axs = plt.subplots(1,2)
fig.tight_layout()
aux=2*65 # number of the considered sample
im=axs[0].imshow(IMSDATA['wavanaly_standardized'][0:0+aux,:],cmap='plasma')
im.set_norm(mynorm)
axs[0].set_xticklabels(IMSDATA['nodes'])
axs[0].set_xlabel('Features')
axs[0].set_ylabel('n° of record')
axs[0].set_title('Normal Functioning')

start=np.shape(IMSDATA['wavanaly_standardized'])[0]-aux
im=axs[1].imshow(IMSDATA['wavanaly_standardized'][start:start+aux,:],cmap='plasma')
im.set_norm(mynorm)
axs[1].set_xticklabels(IMSDATA['nodes'])
axs[1].set_yticklabels(np.arange(start,start+aux))
axs[1].set_xlabel('Features')
axs[1].set_ylabel('n° of record')
axs[1].set_title('Abnormal Functioning')

cb=fig.colorbar(im)
cb.set_label('Power of the feature')
plt.show()
# %%
