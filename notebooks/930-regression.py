from msilib import Feature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import src

src.vis.set_matplotlib_params()
# %% generate data
n_samples=400
x=np.linspace(-2,4,n_samples)
noise=np.random.normal(0,1,n_samples)

y=2*np.exp(x)-1.5*x**3+1.3*np.cos(x)+noise

# %% plot data
fig, ax = plt.subplots()
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# %% matrix form
A=np.zeros((len(x),3))
for row, xi in enumerate(x):
    A[row,0]=np.exp(xi)
    A[row,1]=xi**3
    A[row,2]=np.cos(xi)

# %% fit model
theta = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),y)

yhat = np.array([theta[0]*np.exp(xi)+theta[1]*xi**3+theta[2]*np.cos(xi) for xi in np.linspace(-2,4,1000)])

ax.plot(np.linspace(-2,4,1000),yhat,c='red',label='Regression line')

ax.scatter(x,y,s=1,c='black', label='Data points')

ax.legend(loc='upper center')
# plt.show()

# in more dimensions:
noise=np.random.normal(0,1,n_samples)
y1 = np.sin(x) + np.cos(x)**3 + noise

Y = np.array([y,y1]).transpose()

Aa = np.zeros((len(x),5))
for row, xi in enumerate(x):
    Aa[row,0]=np.exp(xi)
    Aa[row,1]=xi**3
    Aa[row,2]=np.cos(xi)
    Aa[row,3]=np.sin(xi)
    Aa[row,4]=np.cos(xi)**3

THETA = np.dot(np.dot(np.linalg.inv(np.dot(Aa.transpose(),Aa)),Aa.transpose()),Y)

Yhat = np.array([THETA[3,1]*np.sin(xi)+THETA[4,1]*np.cos(xi)**3 for xi in np.linspace(-2,4,1000)])


# %% plot data
fig = plt.figure('figure1')#,figsize=[15, 15])

ax = plt.axes(projection='3d')
print("THETA: ", THETA)
print("shape of x: ", x.shape)
print("shape of yhat: ", yhat.shape)
print("shape of Yhat: ", Yhat.shape)

ax.scatter(x,y,y1,s=1,c='black', label='Data points')
ax.plot(np.linspace(-2,4,1000),yhat,Yhat,c='red',label='Regression line')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 3")

plt.show()
# %%
