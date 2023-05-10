import pandas as pd
import numpy as np
import seaborn as sb
import os

# folder path
dirPath = "./data/raw/1st_test_IMSBearing/"
fileName = "2003.10.22.12.06.24"

rawData=pd.read_csv(dirPath+fileName,delimiter='\t',names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"])
print('\n'+rawData.head)