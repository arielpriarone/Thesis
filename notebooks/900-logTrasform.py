from cProfile import label
from turtle import color
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
from rich import print  
import src
import matplotlib as mpl
import matplotlib as mpl
src.vis.set_matplotlib_params()

# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

x = np.logspace(np.log(0.0000000001), np.log(51), 500)-1

y = -np.log(x+1-10**-6)

fig, ax = plt.subplots()
ax.plot(x, y, label=r'$y = -\log(x+1-10^{-6})$',color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True, which='both')
ax.axvline(x=-1, color='k', linestyle='--', label='x=-1')
ax.set_xlim(-1.8, 50)
ax.legend()

plt.show()