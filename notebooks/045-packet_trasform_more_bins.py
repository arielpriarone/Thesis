# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import src
import importlib
import pywt
_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

samplFreq   = 2000 # Hz
Tend        = 1 #end time
time        = np.arange(0, Tend, 1/samplFreq) # Time points
freq        = np.array([2,53,70,200,400])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range) # this is the disturbance

sigUndisturbed = 2*sin1+5*sin2+sin3+1*sin4

# %%
# wavelet packet
wp = pywt.WaveletPacket(data=sigUndisturbed, wavelet='db10', mode='symmetric',maxlevel=6)
nodes=[node.path for node in wp.get_level(wp.maxlevel, 'natural')]
print(nodes)

# %%
# reconstruct the original signal by decomposition
new_wp = pywt.WaveletPacket(data=None, wavelet='db10', mode='symmetric')

for index in nodes:
    new_wp[index]=wp[index].data
    print(index)
    print(wp[index].data)
powers=[np.linalg.norm(wp[index].data) for index in nodes]
#print(new_wp.reconstruct(update=True))
#print(sigUndisturbed-new_wp.data)


# %%
# # plotting

# fig, axes = plt.subplots(1+2**wp.maxlevel,sharex=True)

# axes[0].plot(time, sigUndisturbed, alpha=1,label='Original signal', color='blue', linewidth=0.4)
# axes[0].plot(time,new_wp.data, alpha=1,label='Reconstructed signal', color='red', linewidth=0.4, linestyle=(0, (5, 5)))
# axes[0].set_xlim(0,Tend)
# axes[0].set_ylabel('Signal')
# axes[0].legend()
# i=0

# for indx in nodes:
#     axes[i+1].plot(np.linspace(0, Tend, int(samplFreq*Tend/(2**wp.maxlevel))),wp[nodes[i]].data, linewidth=0.4)
#     axes[i+1].set_ylabel(indx)
#     i+=1

# axes[2**wp.maxlevel].set_xlabel('Time [s]')
# fig.align_ylabels(axes)
# plt.show()


# %%
fig, ax = plt.subplots()
ax.bar(nodes,powers)
ax.tick_params(axis='x',rotation=90)
plt.show()
