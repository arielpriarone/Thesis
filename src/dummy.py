import numpy as np

a = np.array([[[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], [[4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]]])
print(a.shape)  # (2, 3, 4)

b = np.array([[7, 7, 7], [8, 8, 8]])
print(b.shape)  # (2, 3)

c = np.concatenate((a, b[...,None]), axis=2)  # ValueError: all the input arrays must have same number of dimensions 
print(c.shape)  # wanted result: (2, 3, 5)