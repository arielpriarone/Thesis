# %%
import importlib # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line
import src
import importlib
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileName= '2003.10.22.12.06.24'

snap=src.data.snapshot()
snap.readImsFile(path=dirPath+fileName, variables="Bearing 3 x")
samplFreq=20000 #hz

# %%
# plotting
#plt.figure(figsize=(10,6))
fig, axes = plt.subplots(nrows=2, ncols=1)

for ax in axes.flat:
    ax.set_axisbelow(True)

axes[0].set_ylabel('Amplitude')
axes[1].set_ylabel('Magnitude')
#axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()
#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc=None)
axes[0].plot(snap.rawData['time'], prepSignal, label='No preprocessing', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5)
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc='Flip')
axes[0].plot(snap.rawData['time'], prepSignal, label='Flip and Reverse', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5)
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc='Hann')
axes[0].plot(snap.rawData['time'], prepSignal, label='Hann window', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5)
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(snap.rawData["Bearing 3 x"],samplFreq,preproc='Hamming')
axes[0].plot(snap.rawData['time'], prepSignal, label='Hamming window', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5)


fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2)
plt.subplots_adjust(top=0.9)
plt.show()
# %%
