# %%
import matplotlib
import matplotlib.font_manager
matplotlib.use('Qt5Agg')
import importlib
import pywt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

src.visualization.set_matplotlib_params()
    
# script settings
dirPath     = auxpath + "./data/raw/1st_test_IMSBearing/"   # folder path
savepath    = os.path.join(auxpath + "./data/processed/", "wavanaly_standardized.pickle") #file to save the analisys
tickpath    = os.path.join(auxpath + "./reports/tickz/")    #file to save the tickz

decompose   = False                                         # decompose using wavelet packet / reload previous decomposition
IMSDATA={}                                                  # empty dictionary to save data 
n_split     = 500                                          # number of sample to split the dataset
timestamps  = []                                            # timestamps of the samples
W = 4.7 # inches, s/3 aspect ratio



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
# ax.set_xlabel('Features')
# ax.set_ylabel('Snapshots')
# ax.set_zlabel('Amplitude')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
ax.zaxis.set_ticks([])
#ax.yaxis.set_ticks(np.arange(0,6,1))
dummy=np.round(np.linspace(0, len(fileList) - 1, 6)).astype(int).tolist()
print(dummy)
#ax.set_yticklabels([fileList[i] for i in dummy])
# Create surface plot
ax.scatter(X, Y, IMSDATA['wavanaly_standardized'], marker='.',c=IMSDATA['wavanaly_standardized']/np.max(IMSDATA['wavanaly_standardized']), cmap='turbo')



# %%
# analyze just one snapshot
aux=10 # number of the considered sample

fig, ax = plt.subplots()
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly'][aux,:], color='gray')
ax.set_xlabel('Features')
ax.set_ylabel('Value')
ax.tick_params(axis='x',rotation=90)
mp2tk.save(tickpath+'WT_SingSnap_a.tex',axis_height='3cm', axis_width='0.9\linewidth')

fig, ax = plt.subplots()
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly'][aux,:]/np.linalg.norm(IMSDATA['wavanaly'][aux,:]))
ax.tick_params(axis='x',rotation=90)
ax.set_ylabel('Amplitude')
mp2tk.save(tickpath+'WT_SingSnap_b.tex',axis_height='3cm', axis_width='294.76926pt')

fig, ax = plt.subplots()
ax.bar(IMSDATA['nodes'],IMSDATA['wavanaly_standardized'][aux,:])
ax.set_ylabel('Amplitude')
ax.tick_params(axis='x',rotation=90)
mp2tk.save(tickpath+'WT_SingSnap_c.tex',axis_height='3cm', axis_width='294.76926pt')




#%%
# print as a heatmap
aux=64 # number of the considered sample
start=np.shape(IMSDATA['wavanaly_standardized'])[0]-aux
mynorm = plt.Normalize(vmin=np.min(IMSDATA['wavanaly_standardized']), vmax=np.max(IMSDATA['wavanaly_standardized']))

# Create some example data (replace these with your own image data)
image1 = IMSDATA['wavanaly_standardized'][0:0+aux,:]
image2 = IMSDATA['wavanaly_standardized'][start:start+aux,:]

# Create a figure and a grid for the layout
fig = plt.figure()
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])  # Add an extra column for the colorbar

# Create the first subplot for image1
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(image1, cmap='plasma', aspect='auto')
im1.set_norm(mynorm)
locator=src.vis.custom_tick_locator(6,IMSDATA['nodes'])
ax1.xaxis.set_major_locator(ticker.FixedLocator(locator))
ax1.set_xticklabels([IMSDATA['nodes'][i] for i in locator], rotation=45)
locator=src.vis.custom_tick_locator(8,np.arange(0,0+aux))
ax1.yaxis.set_major_locator(ticker.FixedLocator(locator))
ax1.set_yticklabels(locator)
ax1.set_xlabel('Features')
ax1.set_ylabel('n$^\circ$ of record')
ax1.set_title('Normal Functioning')

# Create the second subplot for image2
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)  # share axes to ensure equal height and width
im2 = ax2.imshow(image2, cmap='plasma', aspect='auto')
im2.set_norm(mynorm)
locator=src.vis.custom_tick_locator(4,IMSDATA['nodes'])
ax2.xaxis.set_major_locator(ticker.FixedLocator(locator))
ax2.set_xticklabels([IMSDATA['nodes'][i] for i in locator], rotation=45)
locator=src.vis.custom_tick_locator(8,np.arange(start,start+aux))
ax2.yaxis.set_major_locator(ticker.FixedLocator(locator))
ax2.set_yticklabels(np.arange(start,start+aux)[i] for i in locator)
ax2.set_xlabel('Features')
ax2.set_ylabel('n$^\circ$ of record')
ax2.set_title('Abnormal Functioning')

# Add a colorbar on the right that has the same height as the two plots
cax = fig.add_subplot(gs[0, 2])
cbar = plt.colorbar(im2, cax=cax)
cbar.set_label('Features value')

# Adjust the layout to prevent overlapping of labels
plt.tight_layout()
# save the plot
mp2tk.save(tickpath+'WT_heatmap.tex',axis_width='\\thesisAxisWidth',axis_height=None,textsize='\\thesisFigFontsize')
# Show the plot
plt.show()


