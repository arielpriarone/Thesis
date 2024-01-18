from matplotlib.lines import lineStyles
from matplotlib.markers import MarkerStyle
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import heatmap
import src

src.vis.set_matplotlib_params()
# %% generate data
x,y = np.meshgrid(np.linspace(-6,3,100),np.linspace(-4,4,100))
z = (x + 2)**2 + (y - 1)**2 - 5 * np.sin(x) - 3 * np.cos(y)
print("Shape x:",x.shape)
print("Shape y:",y.shape)
print("Shape z:",z.shape)
# %% plot data
fig, ax = plt.subplots()
ax.set_xlabel(r'$\theta_1$')
ax.set_ylabel(r'$\theta_2$')

plot = ax.contourf(x,y,z,levels=100,cmap='viridis',label='Density')
cbar=fig.colorbar(plot)
cbar.ax.set_ylabel('Cost function')


# %% fit model
def grad(x,y):
    return np.array([2*(x+2)-5*np.cos(x),2*(y-1)+3*np.sin(y)])

def grad_descent(x,y,lr=0.01,epochs=20):
    theta = np.array([x,y])
    points = theta
    for i in range(epochs):
        theta = theta - lr*grad(theta[0],theta[1])
        points = np.vstack((points,theta))
    return theta, points

np.random.seed(42)
def sto_grad_descent(x,y,lr=0.01,epochs=20):
    theta = np.array([x,y])
    points = theta
    for i in range(epochs):
        theta = theta - lr*grad(theta[0],theta[1])+np.random.normal(0,0.2,2)
        points = np.vstack((points,theta))
    return theta, points

theta, points = grad_descent(-2,-3,lr=0.1,epochs=20)

ax.plot(points[:,0],points[:,1],linewidth=1,marker='.',c='k',label='GD - global minimum')

theta, points = grad_descent(-1,3,lr=0.1,epochs=20)
ax.plot(points[:,0],points[:,1],linewidth=1,marker='.',c='r',label='GD - local minimum')

theta, points = grad_descent(2,-3,lr=0.45,epochs=20)
ax.plot(points[:,0],points[:,1],linewidth=1,marker='.',c='b',label='GD - LR too high')

theta, points = sto_grad_descent(-5,3,lr=0.1,epochs=20)
ax.plot(points[:,0],points[:,1],linewidth=1,marker='.',c='m',label='Stocastic GD')

plt.legend(bbox_to_anchor=(1.35, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

plt.show()