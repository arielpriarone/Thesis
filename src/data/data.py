import pandas as pd
import numpy as np
import seaborn as sb
import os

class snapshot: #this should contain all the useful information about a snapshot (axis, timastamp, features etc...)
    def __init__(self,rawData=None):
        self.rawData=rawData
    def readImsFile(self,path):
        __imsTimeInterval=1 # 1[s] intervals records
        # read a ims formatted file data - ref: The  data  was  generated  by  the  NSF  I/UCR  Center  for  Intelligent  Maintenance  Systems  
        # (IMS  – www.imscenter.net) with support from Rexnord Corp. in Milwaukee, WI. 
        self.rawData=pd.read_csv(path,delimiter='\t',names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"])
        self.rawData.insert(0,"time",np.linspace(0,__imsTimeInterval,int(len(self.rawData.index)))) # linspace prioritize endpoint, arange priorityze increment

if __name__=='__main__': # just for testin, not useful as package functionality
    print(f'the script \"{os.path.basename(__file__)}\" is ruinning as main!')
    dirPath = "./data/raw/1st_test_IMSBearing/"
    fileName = "2003.10.22.12.06.24"
    dummy=snapshot()
    dummy.readImsFile(path=dirPath+fileName)
    print(f"the dataframe has {len(dummy.rawData.index)} rows")
    print(dummy.rawData.head)