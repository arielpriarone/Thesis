import src
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def animate(i, FA : src.features.FA, axs):
    axs = FA.barPlotFeatures(axs)


fig = plt.figure()  # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)  # Add subplot to the main fig window

FA = src.features.FA(r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml")

exit=False
while not exit:
    axs = FA.initialize_barPlotFeatures(ax)
    if axs is not None:
        print("Plot initialized")
        exit=True

# Create a partial function to pass the 'col' argument to animate
animate_partial = partial(animate, FA=FA, axs = ax)

# Matplotlib Animation Function that takes care of real-time plot.
ani = animation.FuncAnimation(fig, animate_partial, cache_frame_data=False, interval=200)  # interval in ms

plt.show()  # Keep Matplotlib plot persistent on the screen until it is closed
