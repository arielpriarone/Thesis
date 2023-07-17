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
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
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
        break
 
 
# %%
fig, ax = plt.subplots()
ax.bar(nodes,powers)
ax.tick_params(axis='x',rotation=90)
plt.show()