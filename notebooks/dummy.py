from re import S
import matplotlib
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from rich import print
matplotlib.use('Qt5Agg')

def f(x,a,b,c)->np.ndarray: #function to fit
    return a*np.exp(b*x)+c

xs=np.random.uniform(0,10,100)
xs.sort()

err=np.random.uniform(-10,10,100) #noise
y=f(xs,2,0.5,0.1)
y_err = y + err

params, cv = opt.curve_fit(f, xs, y_err, p0=[1,1,1]) #fitting
x=np.random.uniform(0,15,100)
x.sort()
y_fit = f(x,*params)

# new procedure
# fit the function a+b*e^(c*x)
# x_f=np.array([-0.99,-0.945,-0.874,-0.859,-0.64,-0.573,-0.433,-0.042,-0.007,0.054,0.088,0.222,0.401,0.465,0.633,0.637,0.735,0.762,0.791,0.981])
# y_f=np.array([0.418,0.412,0.452,0.48,0.453,0.501,0.619,0.9,0.911,0.966,0.966,1.123,1.414,1.683,2.101,1.94,2.473,2.276,2.352,3.544])
x_f=xs.copy()
y_f=y_err.copy()
S=np.array([0])
for k in range(1,len(x_f)):
    S=np.append(S,S[-1]+0.5*(y_f[k]+y_f[k-1])*(x_f[k]-x_f[k-1]))

A=np.linalg.norm(x_f-x_f[0])**2
B=np.sum((x_f-x_f[0])*S)
C=B
D=np.linalg.norm(S)**2
MAT1=np.matrix([[A,B],[C,D]])
print(MAT1)

A=np.sum((y_f-y_f[0])*(x_f-x_f[0]))
B=np.sum((y_f-y_f[0])*S)
MAT2=np.matrix([[A],[B]])
print(MAT2)

RES=MAT1**(-1)*MAT2
print(RES)
A1=RES[0]
B1=RES[1]

a1=-A1/B1
c1=B1.copy()
c2=B1.copy()

theta=np.exp(c2*x_f)

A=x_f.size
B=np.sum(theta)
C=B.copy()
D=np.linalg.norm(theta)**2
MAT1=np.matrix([[A,B],[C,D]])

A=np.sum(y_f)
B=np.sum(y_f*theta.reshape(-1,1))
MAT2=np.matrix([[A],[B]])

RES=MAT1**(-1)*MAT2
A2=RES[0]
B2=RES[1]
params = np.array([B2,c2,A2]).reshape(-1)
print(params)
y_french = f(x,*params) #french method

fig, ax = plt.subplots()
ax.plot(xs,y)
ax.scatter(xs,y_err)
ax.plot(x,y_fit)
ax.plot(x,y_french)



plt.show()


