# %%
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import src
import importlib
import pywt
import matplotlib as mpl
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

src.visualization.set_matplotlib_params()
mpl.rcParams['text.usetex'] = True  
samplFreq   = 400 # Hz
Tend        = 1 #end time
time        = np.arange(0, Tend, 1/samplFreq) # Time points
freq        = np.array([2,5,13,80,400])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range) # this is the disturbance

sigUndisturbed = 2*sin1+5*sin2+3*sin3+1*sin4
# wavelet packet
wp = pywt.WaveletPacket(data=sigUndisturbed, wavelet='db1', mode='symmetric',maxlevel=2)
nodes=[node.path for node in wp.get_level(wp.maxlevel, 'natural')]
print(nodes)

# %%
# reconstruct the original signal by decomposition
new_wp = pywt.WaveletPacket(data=None, wavelet='db1', mode='symmetric')

for index in nodes:
    new_wp[index]=wp[index].data
    print(index)
    print(wp[index].data)
powers=[np.linalg.norm(wp[index].data) for index in nodes]
print(new_wp.reconstruct(update=True))
print(sigUndisturbed-new_wp.data)



# %%
# plotting

fig, ax = plt.subplots(3, 2)
ax[0,0].set_xlabel('Sample [-]')
ax[0,0].set_ylabel('Amplitude [-]')
ax[0,0].plot(sigUndisturbed, color='k', linewidth=0.4)
ax[0,0].set_title('$x(n)$')

ticks=['$|x_{'+node+'}(n)|$' for node in nodes]
ax[0,1].bar(ticks,powers, color='k')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel('Node')
ax[0,1].set_ylabel('Power [-]')
ax[0,1].set_title('Power of the subbands')


ax[1,0].plot(wp['aa'].data,'k', linewidth=0.4)
ax[1,0].set_title('$x_{aa}(n)$')
ax[1,0].set_ylabel('Amplitude [-]')
ax[1,0].set_xlabel('Sample [-]')

ax[1,1].plot(wp['ad'].data,'k', linewidth=0.4)
ax[1,1].set_title('$x_{ad}(n)$')
ax[1,1].set_ylabel('Amplitude [-]')
ax[1,1].set_xlabel('Sample [-]')

ax[2,0].plot(wp['da'].data,'k', linewidth=0.4)
ax[2,0].set_title('$x_{da}(n)$')
ax[2,0].set_ylabel('Amplitude [-]')
ax[2,0].set_xlabel('Sample [-]')

ax[2,1].plot(wp['dd'].data,'k', linewidth=0.4)
ax[2,1].set_title('$x_{dd}(n)$')
ax[2,1].set_ylabel('Amplitude [-]')
ax[2,1].set_xlabel('Sample [-]')

ax[2,0].get_shared_x_axes().join(ax[1,0], ax[2,0])
# ax[1,0].set_xticklabels([])

ax[2,1].get_shared_x_axes().join(ax[1,1], ax[2,1])
# ax[1,1].set_xticklabels([])

ax[2,0].get_shared_y_axes().join(ax[2,1], ax[2,0])
# ax[2,1].set_yticklabels([])

ax[1,0].get_shared_y_axes().join(ax[1,1], ax[1,0])
# ax[1,1].set_yticklabels([])

plt.tight_layout()

plt.show()

# %%
