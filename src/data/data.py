import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime as dt
import os

# folder path
dir_path = "./data/raw/1st_test_IMSBearing/"


# Iterate directory
indx=0
for local_path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, local_path)):
        if indx is 0:   # at first initialize the array
            temp= np.loadtxt(dir_path+local_path,delimiter="\t")
        else:           # if already initialized
            if temp.ndim < 3: # if i want to stack but its 2d array
                temp= temp[...,None] # add a 3rd dimention (magic!)
            temp=np.concatenate((temp, np.loadtxt(dir_path+local_path,delimiter="\t")[...,None]),axis=2)
        indx+=1
print(temp)
# # Iterate directory
# indx=0
# for local_path in os.listdir(dir_path):
#     # check if current path is a file
#     if os.path.isfile(os.path.join(dir_path, local_path)):
#         #read the file
#         temp = pd.read_csv(dir_path+local_path,delimiter="\t",names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y",
#                                                                "Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"])
#         if indx is 0:   # at first initialize the array
#             rawdata = temp
#             keys=str(dt.strptime(local_path, '%Y.%m.%d.%H.%M.%S'))
#         else:           # if already initialized
#             keys=[keys, str(dt.strptime(local_path, '%Y.%m.%d.%H.%M.%S'))]
#             rawdata=pd.concat([rawdata, temp])
#         indx+=1
# print(rawdata)


# rawData=pd.read_csv(dir_path+local_path,delimiter="\t",names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y",
#                                                                "Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"])

# print(rawData.shape)
# print(rawData.head)