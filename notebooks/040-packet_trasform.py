# %%
import numpy as np
import matplotlib.pyplot as plt
import src
import importlib
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

samplFreq   = 5000 # Hz
Tend        = 4 #end time
time        = np.arange(0, Tend, 1/samplFreq) # Time points
freq        = np.array([2,5,7,15,100])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range) # this is the disturbance

sigUndisturbed = 2*sin1+5*sin2+sin3+1*sin4


# plotting
fig, axes = plt.subplots(nrows=2, ncols=1,)

for ax in axes.flat:
    ax.set_axisbelow(True)
    ax.grid(True,'both','both')
    ax.set_ylabel('Magnitude')
    ax.minorticks_on()
#axes[1].set_yscale('log')
axes[0].set_xlabel('Time [s]');axes[1].set_xlabel('Frequency [Hz]')
plt.tight_layout()
axes[0].plot(time, sigUndisturbed, alpha=1,label='Time domain signal', color='blue', linewidth=0.5); axes[0].set_xlim(0,Tend)
plt.show()