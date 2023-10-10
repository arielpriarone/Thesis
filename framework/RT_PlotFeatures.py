import time
import src
import pymongo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


def animate(i, FA : src.features.FA, axs):
    axs = FA.barPlotFeatures(axs)


fig = plt.figure()  # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)  # Add subplot to the main fig window

FA = src.features.FA("../config.yaml")

# Create a partial function to pass the 'col' argument to animate
animate_partial = partial(animate, FA=FA, axs = ax)

# Matplotlib Animation Function that takes care of real-time plot.
ani = animation.FuncAnimation(fig, animate_partial, cache_frame_data=False, interval=500)  # interval in ms

plt.show()  # Keep Matplotlib plot persistent on the screen until it is closed
