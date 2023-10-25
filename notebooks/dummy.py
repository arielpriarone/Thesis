import matplotlib
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from rich import print
matplotlib.use('Qt5Agg')

def f(x,a,b,c,d): #function to fit
    return a*np.exp(b*x)+c

xs=np.random.uniform(0,10,100)
xs.sort()

err=np.random.uniform(-10,10,100) #noise
y=f(xs,1,0.5,0.1,0)
y_err = y + err

params, cv = opt.curve_fit(f, xs, y_err, p0=[1,1,1,1]) #fitting

x=np.random.uniform(0,15,100)
x.sort()
y_fit = f(x,*params)

fig, ax = plt.subplots()
ax.plot(xs,y)
ax.scatter(xs,y_err)
ax.plot(x,y_fit)
plt.show()


