import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment

# Set the parameters - first wave
duration = 60*10  # Duration of the audio in seconds
sampling_rate = 44100  # Number of samples per second
frequencies =   [30,   70,  100, 300,   800,   1400]  # Base frequency of the waveform
amplitudes =    [0.1,  1.0,  1.0,  0.1,  1.0,    1.0]  # Amplitudes of the harmonics
frequencies =   [30,   70,  100, 300,   800]  # Base frequency of the waveform
amplitudes =    [2.0,  3.0,  2.0,  2.0,  1.0]  # Amplitudes of the harmonics
frequencies =   [80,   120]  # Base frequency of the waveform
amplitudes =    [1.0,  3.0]  # Amplitudes of the harmonics
frequencies =   [10,   30,  60]  # Base frequency of the waveform
amplitudes =    [1,  1.0,  0.1]  # Amplitudes of the harmonics
# frequencies =   [30,   70,  100,   800,   1400]  # Base frequency of the waveform
# amplitudes =    [0.1,  1.0,  1.0,    1.0,    1.0]  # Amplitudes of the harmonics
# frequencies =   [30,   70,  100,   800,   1400]  # Base frequency of the waveform
# amplitudes =    [0.1,  0.8,  1.0,    3.0,    0.6]  # Amplitudes of the harmonics

# Generate the time axis
t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)

# Generate the waveform
waveform = np.zeros_like(t)
for amplitude, frequency in zip(amplitudes, frequencies):
    waveform += amplitude * np.sin(2 * np.pi * frequency * t)

# Normalize the waveform
waveform /= np.max(np.abs(waveform))

# Save the waveform as a .wav file
wav.write('data/output_5.wav', sampling_rate, waveform)


