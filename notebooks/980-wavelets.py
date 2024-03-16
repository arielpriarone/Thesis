from pyrsistent import b
import pywt
import src
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
src.visualization.set_matplotlib_params()
mpl.rcParams['text.usetex'] = True

t=np.linspace(-5,5,1000)
j = 1j #create complex number
fb=1 #decay of the wavelet
fc=1 #frequency of the wavelet
print(j)
PHI = 1/np.sqrt(np.pi*fb)*np.exp(j*2*np.pi*fc*t)*np.exp(-t**2/fb)

fig, ax = plt.subplots(1, 2,sharey=True)
ax[0].plot(t, np.real(PHI),'k')
ax[0].set_title('Real part')
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(t, np.imag(PHI),'k')
ax[1].set_title('Imaginary part')
ax[1].set_xlabel('Time')


linestyles = ['-', '--', '-.', ':']
fig, ax = plt.subplots(1, 2,sharey=True)
for i, (a,b) in enumerate([(1,0),(1,3),(0.5,-3),(5,0)]):
    tp=(t-b)/a
    PHI = 1/np.sqrt(np.pi*fb)*np.exp(j*2*np.pi*fc*tp)*np.exp(-tp**2/fb)
    ax[0].plot(t, np.real(PHI),'k',linestyle=linestyles[i],label='$\psi_{'+str(a)+','+str(b)+'}(t)$')
    ax[0].set_title('Real part')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(t, np.imag(PHI),'k',linestyle=linestyles[i])
    ax[1].set_title('Imaginary part')
    ax[1].set_xlabel('Time')

fig.legend(loc='upper center',bbox_to_anchor=(0.5,1),ncol=4)


fig, ax = plt.subplots(1, 2)
import numpy as np
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow

pi = np.pi
v1, v2, v3 = 64, 128, 32

#%%# Helper fn + params #####################################################
def exp_am(t, offset):
    return np.exp(-pi*((t - offset) / .1)**10)

#%%# Make `x` & plot #########################################################
t = np.linspace(0, 1, 2048, 1)
x = (np.sin(2*pi * v1 * t) * exp_am(t, .2) +
     (np.sin(2*pi * v1 * t) + 2*np.cos(2*pi * v2 * t)) * exp_am(t, .5)  + 
     (2*np.sin(2*pi * v2 * t) - np.cos(2*pi * v3 * t)) * exp_am(t, .8))
plot(x, title="x(t) | t=[0, ..., 1], %s samples" % len(x), show=1)

#%%# Take CWT & plot #########################################################
Wx, scales = cwt(x, 'morlet')
imshow(Wx, yticks=scales, abs=1,
       title="abs(CWT) | Morlet wavelet",
       ylabel="scales", xlabel="samples")

plt.show()
