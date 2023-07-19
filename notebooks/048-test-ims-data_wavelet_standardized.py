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
auxpath='' # auxilliary path because interactive mode treat path differently 
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
    auxpath='.'
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
stdsclr=StandardScaler()
# folder path
dirPath = auxpath + "./data/raw/1st_test_IMSBearing/"
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
        powers=[np.linalg.norm(wp[index].data) for index in nodes]  # getting the powers of the wavelet coefs
        if indx==0:
            wavanaly=powers
        else:
            wavanaly = np.vstack([wavanaly, powers])
        if indx==1500: #split training dataset and validation dataset
            stdsclr.fit(wavanaly) 
        indx+=1
        print(indx)
wavanaly_standardized=stdsclr.transform(wavanaly)

#%%
from mpl_toolkits import mplot3d
x = np.arange(0,wavanaly_standardized.shape[1],1)
y = np.arange(0,wavanaly_standardized.shape[0],1)
X,Y = np.meshgrid(x,y)
# Plot a 3D surface
print(X.shape)
print(Y.shape)
print(wavanaly_standardized.shape)
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
ax.scatter(X, Y, wavanaly_standardized, marker='.',c=wavanaly_standardized/np.max(wavanaly_standardized), cmap='turbo')
ax.set_xticklabels(nodes)
plt.show()

# %%
# analyze just one snapshot
fig, axs = plt.subplots(3,1,sharex=True)
fig.tight_layout()
aux=200 # number of the considered sample
axs[0].bar(nodes,wavanaly[aux,:])
axs[0].tick_params(axis='x',rotation=90)
axs[0].set_title('single snapshot, raw')

axs[1].bar(nodes,wavanaly[aux,:]/np.linalg.norm(wavanaly[aux,:]))
axs[1].tick_params(axis='x',rotation=90)
axs[1].set_title('single snapshot, power normalized')

axs[2].bar(nodes,wavanaly_standardized[aux,:]/np.linalg.norm(wavanaly_standardized[aux,:]))
axs[2].tick_params(axis='x',rotation=90)
axs[2].set_title('single snapshot, standardized along features axes')
plt.show()


# %%
