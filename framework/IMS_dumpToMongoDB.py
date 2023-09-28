import os
import src
import importlib
import numpy as np

_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

dirPath=r'C:\Users\ariel\Documents\Courses\Tesi\Code\data\raw\1st_test_IMSBearing'


# dump all the files in the folder to mondoDB
fileList = os.listdir(dirPath)                                              # al files in folder
for fileName in fileList:
    # check if current path is a file
    path=os.path.join(dirPath, fileName) # complete path including filename
    if os.path.isfile(path):
        src.data.IMS_to_mongo(database='IMS',collection='Raw',
                              filePath= path, n_of_test=1, # this folder os first test
                                sensors=['Bearing 1 x', 'Bearing 1 y'])


