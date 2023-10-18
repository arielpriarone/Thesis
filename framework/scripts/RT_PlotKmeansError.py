import src
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def animate(i, PLTR : src.vis.Plotter, axs):
    axs = PLTR.plot_Kmeans_error(axs)


fig = plt.figure()  # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)  # Add subplot to the main fig window

PLTR = src.vis.Plotter(r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml")

# Create a partial function to pass the 'col' argument to animate
animate_partial = partial(animate, PLTR=PLTR, axs = ax)

# Matplotlib Animation Function that takes care of real-time plot.
ani = animation.FuncAnimation(fig, animate_partial, cache_frame_data=False, interval=20)  # interval in ms

plt.show()  # Keep Matplotlib plot persistent on the screen until it is closed