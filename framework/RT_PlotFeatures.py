import time
import src
import pymongo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


def animate(i, col: pymongo.collection.Collection):
    snap = col.find().sort('timestamp', pymongo.DESCENDING).limit(1)[0]  # latest document in collection
    x = (snap['timestamp'])  # Append x value
    y = (snap['Bearing 1 x']["timeSerie"])  # Append y value

    ax.clear()  # Clear last data frame
    ax.plot(y)  # Plot new data frame

fig = plt.figure()  # Create Matplotlib plots fig is the 'higher level' plot window
ax = fig.add_subplot(111)  # Add subplot to the main fig window

DB_man = src.data.DB_Manager("../config.json")

# Create a partial function to pass the 'col' argument to animate
animate_partial = partial(animate, col=DB_man.col_raw)

# Matplotlib Animation Function that takes care of real-time plot.
ani = animation.FuncAnimation(fig, animate_partial, cache_frame_data=False, interval=100)

plt.show()  # Keep Matplotlib plot persistent on the screen until it is closed
