from click import style
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import line

phi     = np.linspace(-180, 180, 36*2+1)
print(phi)
theta   = phi.copy()/180*np.pi

fig, ax = plt.subplots()
ax.set_aspect('equal')
theta_ = np.linspace(-180, 180, 100)/180*np.pi
for phi_ in phi:
    linewidth = 0.5 if np.mod(phi_,10) < 1 else 0.7
    phi_ = phi_/180*np.pi
    ax.plot(np.sin(theta_), np.cos(theta_)*np.sin(phi_), color = 'k', linewidth=linewidth)
    ax.plot(np.cos(theta_)*np.sin(phi_), np.sin(theta_), color = 'k', linewidth=linewidth)

# hide the axes and ticks
ax.set_frame_on(False)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()




