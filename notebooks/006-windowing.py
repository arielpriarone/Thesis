from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import src
src.visualization.set_matplotlib_params()   

# Generate the window functions
window_length = 100
hann_window = np.hanning(window_length)
hamming_window = np.hamming(window_length)

# Plot the window functions
plt.plot(hann_window, label='Hann', linestyle='-', color='k')
plt.plot(hamming_window, label='Hamming', linestyle='--', color='k')
plt.xlabel(r'$n$')
plt.ylabel('Amplitude')
plt.xticks([0, 25, 50, 75, 100], ['1', r'$\frac{1}{4}N$', r'$\frac{1}{2}N$', r'$\frac{3}{4}N$', r'$N$'])
plt.legend()
plt.show()
