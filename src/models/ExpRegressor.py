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