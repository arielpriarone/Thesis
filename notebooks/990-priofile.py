from matplotlib.lines import lineStyles
import numpy as np
from rich import print
import matplotlib.pyplot as plt
import matplotlib as mpl
import src.visualization.vis as vis
import matplotlib.ticker as ticker
vis.set_matplotlib_params()
mpl.rcParams['text.usetex'] = True  

t_step=1e-3 # s
v = 0.0 # start from rest

acc = 1 # m/s^2
max_spd = 0.5 # m/s

def done(x, ref, window=300):
    done = True
    for i in range(np.minimum(window, len(x))):
        if (x[-i-1] < ref-0.01) or (x[-i-1] > ref+0.01):
            done = False
    return done

fig, ax = plt.subplots()
linestles = ['-', '--', '-.', ':']
i = 0
for (max_spd, acc) in zip([0.8,0.4,0.4,0.8],[6,3,6,8]):
    x = np.array([-0.4])
    for ref in np.array([-0.2, 0.0, 0.2, 0.4, -0.4, -0.2, 0.0, 0.2, 0.4, -0.4]):
        
        while not done(x, ref):
            v_ref = np.sqrt(2*acc*np.abs(ref-x[-1]))
            if ref < x[-1]:
                v_ref = -v_ref
            if v_ref > max_spd:
                v_ref = max_spd
            if v_ref < -max_spd:
                v_ref = -max_spd
            if v < v_ref:
                v += acc*t_step
            if v > v_ref:
                v -= acc*t_step
            x = np.append(x, x[-1]+v*t_step)
    print(f"ref={ref}, acc={acc}, max_spd={max_spd}, x={x}")
    ax.plot(x*1000, label=f"$a={acc}, v={max_spd}$", linestyle=linestles[i], color='black')
    i += 1
ax.legend(ncol=4, loc='upper center outside')
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Position [mm]")
ax.xaxis.set_major_locator(ticker.AutoLocator())
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

plt.show()