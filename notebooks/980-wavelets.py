import pywt
import src
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
src.visualization.set_matplotlib_params()

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
plt.show()
