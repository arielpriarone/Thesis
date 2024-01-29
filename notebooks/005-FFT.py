import arrow
import numpy as np
import src

src.visualization.set_matplotlib_params()
import matplotlib.pyplot as plt
import importlib

_ = importlib.reload(
    src
)  # this make changes in the src package immediately effective without restarting the kernel
from IPython import get_ipython

if src.visualization.isNotebook():  # run widget only if in interactive mode
    get_ipython().run_line_magic("matplotlib", "widget")
from rich import print
from scipy import fft, arange


def plotSpectrum(y, Fs):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    n = len(y)  # length of the signal
    k = arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(round(n / 2))]  # one side frequency range

    Y = fft.fft(y) / n  # fft computing and normalization
    Y = Y[range(int(np.round(n / 2)))]

    time = np.linspace(0, T, n)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(time,y,'k')
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    ax[1].scatter(frq, abs(Y),marker='.', color = 'k')  # plotting the spectrum
    ax[1].set_xlabel("Freq [Hz]")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_yscale('log')
    ax[1].set_ylim([1e-4, 1e1])
    plt.tight_layout()
    plt.tight_layout()
    return fig, ax


samplFreq = 1000  # sampling frequency
freqs = [2, 5, 7, 15]  # frequencies of the known armonics
periods = [1 / f for f in freqs]  # periods of the known armonics
time = np.arange(0, 1, 1 / samplFreq)  # time vector

# creating the signal
signal = np.zeros(len(time))
for freq in freqs:
    signal += np.sin(2 * np.pi * freq * time)


print(f"periods: {periods}")

# plotting the signal and the spectrum
fig, ax = plotSpectrum(signal, samplFreq)
ax[1].annotate('known armonics', xy=(30,0.5), xytext=(70, 1), xycoords='data',arrowprops=dict(arrowstyle='->', color='k'))

# injecting disturbance
for t in time:
    if 0.59<t<0.61:
        signal[np.where(time==t)] += np.sin(2 * np.pi * 50 * (t-0.59))


# plotting the signal and the spectrum
fig, ax = plotSpectrum(signal, samplFreq)
ax[0].annotate('disturbance', xy=(0.6,0.4), xytext=(0.6, 0.05), xycoords='axes fraction', 
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='-[, widthB=0.4, lengthB=1', color='k'))
ax[1].annotate('disturbance', xy=(60,0.02), xytext=(60, 0.1), xycoords='data',arrowprops=dict(arrowstyle='->', color='k'))


# plot without synchronization
signal = np.zeros(len(time)) # original signal
for freq in freqs:
    signal += np.sin(2 * np.pi * freq * time)
fig, ax = plotSpectrum(signal[np.where(time<0.9)], samplFreq)
ax[1].annotate('spectral leakage', xy=(24,0.053), xytext=(70, 1), xycoords='data',arrowprops=dict(arrowstyle='->', color='k'))

plt.show()