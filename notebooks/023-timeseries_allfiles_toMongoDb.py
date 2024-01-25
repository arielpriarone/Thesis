# %%
import importlib
import time
from bson import Timestamp # COMMENTO AGGIUNTO
import matplotlib.pyplot as plt
import numpy as np
import src
import os
import importlib
from rich import print
from datetime import datetime
from  pymongo import MongoClient

from src.data.data import mongoConnect

_ = importlib.reload(src)   # this make changes in the src package immediately effective without restarting the kernel

from IPython import get_ipython
if src.visualization.isNotebook(): # run widget only if in interactive mode
    get_ipython().run_line_magic('matplotlib', 'widget')
src.vis.set_matplotlib_params()

sensors = ["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y", "Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"]

# RMS
def rms(x):
    return np.sqrt(np.mean(x**2))

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileNames = os.listdir(dirPath)
snap=src.data.snapshot()
client = MongoClient("mongodb://localhost:27017/")
db = client["BACKUP"]
db.create_collection("RawData_1st_test_IMSBearing")
collection = db["RawData_1st_test_IMSBearing"]

for i, fileName in enumerate(fileNames):
    data = {}
    # check if current path is a file
    if os.path.isfile(os.path.join(dirPath, fileName)):
        print(f"Reading file {i+1}/{len(fileNames)}: {fileName}")
        data["timestamp"] = src.data.IMS_filepathToTimestamp(fileName)
        snap.readImsFile(path=dirPath+fileName, variables=sensors)
        for sensor in sensors:
            data[sensor] = list(snap.rawData[sensor])
        collection.insert_one(data)

            

        

