import numpy as np
def ExpRegressor(x,y):
    """
    Fits the function a*exp(b*x)+c to the given data points.
    https://it.scribd.com/doc/14674814/Regressions-et-equations-integrales

    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.

    Returns:
    list: A list containing the fitted parameters [a,b,c].
    """

    # the original procedure fits a+b*e^(c*x): 
    x_f=np.array(x)
    y_f=np.array(y)
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
    return np.array([B2,c2,A2]).reshape(-1).tolist()

import scipy.optimize as opt
def f(x,a,b,c): #function to fit for novelty prediction
        return a*np.exp(b*x)+c

if __name__ == '__main__':
   
    import src
    import matplotlib.pyplot as plt
    src.visualization.set_matplotlib_params()
    import matplotlib as mpl

    fig_width = mpl.rcParams['figure.figsize'][0]
    fig_height = mpl.rcParams['figure.figsize'][1]
    mpl.rcParams['figure.figsize'] = (fig_width, 0.5 * fig_height) # Set figure size to 1:2 ratio
    mpl.rcParams["text.usetex"] = True

    

    # Generate data

    x = np.linspace(-10, 5, 300)
    noise = np.random.normal(0, 1, x.size)

    y = 2 * np.exp(0.5 * (x+0.5*noise)) + 1 + noise

    y1 = np.zeros(y.size)
    for idx, val in enumerate(y):
        y1[idx] = y[idx]
        if idx >= x.size/3:
            y1[idx] = y[idx] + 10

    fig, ax = plt.subplots()
    ax.scatter(x[0:int(np.floor(x.size/3))], y[0:int(np.floor(x.size/3))], marker='.', color='black', label='Normal')
    ax.scatter(x[int(np.ceil(x.size/3)):], y[int(np.ceil(x.size/3)):], marker='.', color='red', label='Anomaly to fit')
    ax.set_xticklabels([])
    ax.set_ylabel('Novelty\n metric')
    ax.legend()
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax.scatter(x[0:int(np.floor(x.size/3))], y1[0:int(np.floor(x.size/3))], marker='.', color='black', label='Normal')
    ax.scatter(x[int(np.ceil(x.size/3)):], y1[int(np.ceil(x.size/3)):], marker='.', color='red', label='Anomaly to fit')
    ax.set_xlabel('Time')
    ax.set_ylabel('Novelty\n metric')
    ax.legend()
    fig.tight_layout()


    x_fit = x[int(np.ceil(x.size/3)):]
    y_fit = y1[int(np.ceil(x.size/3)):]

    a,b,c = ExpRegressor(x_fit, y_fit) #Fits the function a*exp(b*x)+c to the given data points.

    y_pred = a * np.exp(b * np.linspace(-10,10,300)) + c

    fig, ax = plt.subplots()
    ax.scatter(x[int(np.ceil(x.size/3)):], y1[int(np.ceil(x.size/3)):], marker='.', color='red', label='Anomaly to fit')
    ax.plot(np.linspace(-10,10,300), y_pred, label='Closed form fitting', color='black')
    ax.set_xlabel('Time')
    ax.set_ylabel('Novelty\n metric')

    fig.tight_layout()

    params, cv = opt.curve_fit(f, x, y) #fitting
    y_pred = f(np.linspace(-10,10,300), *params)
    ax.plot(np.linspace(-10,10,300), y_pred, label='scipy fitting', color='blue')
    ax.legend()

    plt.show()