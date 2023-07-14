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

samplFreq   = 20000 # Hz
Tend        = 1 #end time
time        = np.arange(0, Tend, 1/samplFreq) # Time points
freq        = np.array([2,5,7,200,400])
sin1        = np.sin(2*np.pi*freq[0]*time)
sin2        = np.sin(2*np.pi*freq[1]*time)
sin3        = np.sin(2*np.pi*freq[2]*time)
sin4        = np.sin(2*np.pi*freq[3]*time)
sin5_range  = np.arange(0,1.5/freq[4], 1/samplFreq)
sin5        = np.sin(2*np.pi*freq[4]*sin5_range) # this is the disturbance

sigUndisturbed = 2*sin1+5*sin2+sin3+1*sin4

# %%
# wavelet packet
wp = pywt.WaveletPacket(data=sigUndisturbed, wavelet='db1', mode='symmetric',maxlevel=3)
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

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(1+2**wp.maxlevel,2)

ax = fig.add_subplot(gs[0, 0])
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [-]')


ax.plot(time, sigUndisturbed, alpha=1,label='Original signal', color='blue', linewidth=0.4)
ax.plot(time,new_wp.data, alpha=1,label='Reconstructed signal', color='red', linewidth=0.4, linestyle=(0, (5, 5)))
ax.set_xlim(0,Tend)
ax.legend()

ax = fig.add_subplot(gs[0, 1])
ax.scatter(nodes,powers)
ax.set_yscale('log')
ax.set_xlabel('Coeficients')
ax.set_ylabel('Power')

i=0
for indx in nodes:
    ax = fig.add_subplot(gs[1+int(i/2), np.mod(i,2)])
    ax.plot(np.linspace(0, Tend, int(samplFreq*Tend/(2**wp.maxlevel))),wp[nodes[i]].data, linewidth=0.4)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Amplitude [-]')
    i+=1

plt.show()


# %%
