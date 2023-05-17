# %%
import numpy as np
import matplotlib.pyplot as plt
import src
import importlib
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

samplFreq   = 10000 # Hz
time        = np.arange(0, 5, 1/samplFreq) # Time points
freq        = np.array([2,5,7,15,100])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range)
aux         = np.zeros([10*samplFreq])
ampUndisturbed = 2*sin1+5*sin2+sin3+1*sin4

# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)

for ax in axes.flat:
    ax.set_axisbelow(True)
    ax.grid(True,'both','both')
    ax.set_ylabel('Magnitude')
    ax.minorticks_on()
axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()

#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc=None)
axes[0].plot(time, prepSignal, alpha=1,label='No preprocessing', color='blue', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5, alpha=1, color='blue')
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Hann')
axes[0].plot(time, prepSignal, alpha=1,label='Hann window', color='green', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5, alpha=1, color='green')
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Hamming')
axes[0].plot(time, prepSignal, alpha=1,label='Hamming window', color='purple', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5, alpha=1, color='purple')
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Flip')
axes[0].plot(np.linspace(0,2*time[-1],len(prepSignal)), prepSignal, alpha=1,label='Flip and Reverse', color='orange', linewidth=0.5)
axes[1].scatter(freqs, FFT,s=0.5, alpha=1, color='orange')

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=4)
plt.subplots_adjust(top=0.9)
plt.show()
# %%
ampReduced_multiplePeriod=ampUndisturbed[0+1234:10000+1234] # reducing the data to multiple base period interval
time_reduced=time[0+1234:10000+1234]

# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)
for ax in axes.flat:
    ax.set_axisbelow(True)
    ax.grid(True,'both','both')
    ax.set_ylabel('Magnitude')
    ax.minorticks_on()
axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()

#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc=None)
axes[0].plot(time_reduced, prepSignal, alpha=1,label='No preprocessing', color='blue', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='No preprocessing', color='blue', linewidth=0.5)
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Hann')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hann window', color='green', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Hann window', color='green', linewidth=0.5)
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Hamming')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hamming window', color='purple', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Hamming window', color='purple', linewidth=0.5)
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Flip')
axes[0].plot(np.linspace(time_reduced[0],time_reduced[0]+2*(time_reduced[-1]-time_reduced[0]),len(prepSignal)), prepSignal, alpha=1,label='Flip and Reverse', color='orange', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Flip and Reverse', color='orange', linewidth=0.5)

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=4)
plt.subplots_adjust(top=0.9)
plt.show()

# %%
ampReduced_fractionedPeriod=ampUndisturbed[0+1234:12000+1234]
time_reduced=time[0+1234:12000+1234]

# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)
for ax in axes.flat:
    ax.set_axisbelow(True)
    ax.grid(True,'both','both')
    ax.set_ylabel('Magnitude')
    ax.minorticks_on()
axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()

#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc=None)
axes[0].plot(time_reduced, prepSignal, alpha=1,label='No preprocessing', color='blue', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='No preprocessing', color='blue', linewidth=0.5)
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Hann')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hann window', color='green', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Hann window', color='green', linewidth=0.5)
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Hamming')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hamming window', color='purple', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Hamming window', color='purple', linewidth=0.5)
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Flip')
axes[0].plot(np.linspace(time_reduced[0],time_reduced[0]+2*(time_reduced[-1]-time_reduced[0]),len(prepSignal)), prepSignal, alpha=1,label='Flip and Reverse', color='orange', linewidth=0.5)
axes[1].plot(freqs, FFT, alpha=1,label='Flip and Reverse', color='orange', linewidth=0.5)

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=4)
plt.subplots_adjust(top=0.9)
plt.show()
# %%
