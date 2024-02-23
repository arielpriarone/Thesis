# %%
import numpy as np
import matplotlib.pyplot as plt
import src
import importlib
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

src.vis.set_matplotlib_params()

samplFreq   = 1000 # Hz
time        = np.arange(0, 1, 1/samplFreq) # Time points
freq        = np.array([2,5,7,15,100])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range)
aux         = np.zeros([10*samplFreq])
ampUndisturbed = sin1+sin2+sin3+sin4

# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)


axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()

#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc=None)
axes[0].plot(time, prepSignal, alpha=1,label='No preprocessing')
axes[1].scatter(freqs, FFT,s=0.5, alpha=1)
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Hann')
axes[0].plot(time, prepSignal, alpha=1,label='Hann window')
axes[1].scatter(freqs, FFT,s=0.5, alpha=1)
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Hamming')
axes[0].plot(time, prepSignal, alpha=1,label='Hamming window')
axes[1].scatter(freqs, FFT,s=0.5, alpha=1)
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampUndisturbed,samplFreq,preproc='Flip')
axes[0].plot(time, prepSignal, alpha=1,label='Flip and Reverse')
axes[1].scatter(freqs, FFT,s=0.5, alpha=1)

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=2)
plt.subplots_adjust(top=0.9)

# # %%
# ampReduced_multiplePeriod=ampUndisturbed[0+1234:10000+1234] # reducing the data to multiple base period interval
# time_reduced=time[0+1234:10000+1234]

# # plotting
# fig, axes = plt.subplots(nrows=2, ncols=1,)

# axes[1].set_yscale('log')
# axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
# axes[0].set_ylabel('amplitude');axes[1].set_ylabel('Magnitude')
# plt.tight_layout()

# #performing fast fourier trasfotm
# # no preprocessing
# FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc=None)
# axes[0].plot(time_reduced, prepSignal, alpha=1,label='No preprocessing', color='blue')
# axes[1].scatter(freqs, FFT, alpha=1, color='blue', s=0.5)
# # Hann window preprocessing
# FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Hann')
# axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hann window', color='green')
# axes[1].scatter(freqs, FFT, alpha=1, color='green', s=0.5)
# # Hamming window preprocessing
# FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Hamming')
# axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hamming window', color='purple')
# axes[1].scatter(freqs, FFT, alpha=1, color='purple', s=0.5)
# # Flip window preprocessing
# FFT, freqs, prepSignal = src.features.FFT(ampReduced_multiplePeriod,samplFreq,preproc='Flip')
# axes[0].plot(np.linspace(time_reduced[0],time_reduced[0]+2*(time_reduced[-1]-time_reduced[0]),len(prepSignal)), prepSignal, alpha=1,label='Flip and Reverse', color='orange')
# axes[1].scatter(freqs, FFT, alpha=1, color='orange', s=0.5)

# fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=2)
# plt.subplots_adjust(top=0.9)


# %%
ampReduced_fractionedPeriod=ampUndisturbed[np.where(time<0.9)]
time_reduced=time[np.where(time<0.9)]

# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)

axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
axes[0].set_ylabel('Amplitude');axes[1].set_ylabel('Magnitude')
plt.tight_layout()

#performing fast fourier trasfotm
# no preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc=None)
axes[0].plot(time_reduced, prepSignal, alpha=1,label='No preprocessing')
axes[1].scatter(freqs, FFT, alpha=1, s=0.5)
# Hann window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Hann')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hann window')
axes[1].scatter(freqs, FFT, alpha=1, s=0.5)
# Hamming window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Hamming')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Hamming window')
axes[1].scatter(freqs, FFT, alpha=1, s=0.5)
# Flip window preprocessing
FFT, freqs, prepSignal = src.features.FFT(ampReduced_fractionedPeriod,samplFreq,preproc='Flip')
axes[0].plot(time_reduced, prepSignal, alpha=1,label='Flip and Reverse')
axes[1].scatter(freqs, FFT, alpha=1, s=0.5)

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=2)
plt.subplots_adjust(top=0.9)

# %%
plt.show()