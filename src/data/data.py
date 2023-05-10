import pandas as pd
import numpy as np
import seaborn as sb
import os


class snapshot:
    def __init__(self,rawData=None):
        self.rawData=rawData
    def readImsFile(self,path):
        # read a ims formatted file data - ref: The  data  was  generated  by  the  NSF  I/UCR  Center  for  Intelligent  Maintenance  Systems  
        # (IMS  – www.imscenter.net) with support from Rexnord Corp. in Milwaukee, WI. 
        self.rawData=pd.read_csv(path,delimiter='\t',names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"])

