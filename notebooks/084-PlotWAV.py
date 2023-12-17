import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import matplotlib.pyplot as plt
import src

src.vis.set_matplotlib_params()

# Set the parameters - first wave
duration = 0.1  # Duration of the audio in seconds
sampling_rate = 44100  # Number of samples per second
frequencies =   [30,   70,  100, 300,   800,   1400]  # Base frequency of the waveform
amplitudes =    [0.1,  1.0,  1.0,  0.1,  1.0,    1.0]  # Amplitudes of the harmonics

# Generate the time axis
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Generate the waveform
waveform = np.zeros_like(t)
for amplitude, frequency in zip(amplitudes, frequencies):
    waveform += amplitude * np.sin(2 * np.pi * frequency * t)

# Normalize the waveform
waveform1 = waveform / np.max(np.abs(waveform))

# Set the parameters - second wave
frequencies =   [30,   70,  100,   800,   1400]  # Base frequency of the waveform
amplitudes =    [0.1,  1.0,  1.0,    1.0,    1.0]  # Amplitudes of the harmonics

# Generate the time axis
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Generate the waveform
waveform = np.zeros_like(t)
for amplitude, frequency in zip(amplitudes, frequencies):
    waveform += amplitude * np.sin(2 * np.pi * frequency * t)

# Normalize the waveform
waveform2 = waveform / np.max(np.abs(waveform))

# Set the parameters - third wave
frequencies =   [30,   70,  100,   800,   1400]  # Base frequency of the waveform
amplitudes =    [0.1,  0.8,  1.0,    3.0,    0.6]  # Amplitudes of the harmonics

# Generate the time axis
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Generate the waveform
waveform = np.zeros_like(t)
for amplitude, frequency in zip(amplitudes, frequencies):
    waveform += amplitude * np.sin(2 * np.pi * frequency * t)

# Normalize the waveform
waveform3 = waveform / np.max(np.abs(waveform))

# plot the three waveforms
fig, ax = plt.subplots()
ax.plot(t, waveform1, label='Waveform 1')
ax.plot(t, waveform2, label='Waveform 2')
# ax.plot(t, waveform3, label='Waveform 3')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
fig.tight_layout()


# plot the fft of the three waveforms
fig, ax = plt.subplots()
ax.plot(np.fft.rfftfreq(len(waveform1), 1 / sampling_rate), np.abs(np.fft.rfft(waveform1)), label='Waveform 1')
ax.plot(np.fft.rfftfreq(len(waveform2), 1 / sampling_rate), np.abs(np.fft.rfft(waveform2)), label='Waveform 2', linestyle='--')
ax.plot(np.fft.rfftfreq(len(waveform3), 1 / sampling_rate), np.abs(np.fft.rfft(waveform3)), label='Waveform 3', linestyle=':')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Amplitude')
ax.legend()
fig.tight_layout()

plt.show()
