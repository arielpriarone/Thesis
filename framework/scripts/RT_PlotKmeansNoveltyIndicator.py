import src
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial

def animate(i, PLTR : src.vis.Plotter, axs):
    PLTR.plot_Kmeans_error(axs)

fig = plt.figure()  # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)  # Add subplot to the main fig window

PLTR = src.vis.Plotter(r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml",'novelty')

exit=False
axs = None  # Initialize axs to None
while not exit:
    axs = PLTR.plot_Kmeans_error_init(ax)
    if axs is not None:
        print("Plot initialized")
        exit=True

# Create a partial function to pass the 'col' argument to animate
animate_partial = partial(animate, PLTR=PLTR, axs = axs)

# Matplotlib Animation Function that takes care of real-time plot.
ani = animation.FuncAnimation(fig, animate_partial, cache_frame_data=False, interval=200)  # interval in ms
plt.show()  # Keep Matplotlib plot persistent on the screen until it is closed
