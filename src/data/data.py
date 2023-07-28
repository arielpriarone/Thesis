import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
import datetime

class snapshot: #this should contain all the useful information about a snapshot (axis, timastamp, features etc...)
    def __init__(self,rawData=None):
        self.rawData=rawData
    def readImsFile(self,path,variables=None,n_of_test=1):
        __imsTimeInterval=1 # 1[s] intervals records
        # read a ims formatted file data - ref: The  data  was  generated  by  the  NSF  I/UCR  Center  for  Intelligent  Maintenance  Systems  
        # (IMS  – www.imscenter.net) with support from Rexnord Corp. in Milwaukee, WI. 
        match n_of_test:
            case 1:
                __names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"]
            case 2:
                __names=["Bearing 1 ", "Bearing 2", "Bearing 3", "Bearing 4"]
            case 3:
                __names=["Bearing 1 ", "Bearing 2", "Bearing 3", "Bearing 4"]
                
        self.rawData=pd.read_csv(path,delimiter='\t',names=__names)
        for i in self.rawData.columns.values.tolist():
            if i in variables:
                pass
            else: # remove the unwanted culumns
                self.rawData.drop(labels=i,axis=1, inplace=True) # axis 1  are columns
        self.rawData.insert(0,"time",np.linspace(0,__imsTimeInterval,int(len(self.rawData.index)))) # linspace prioritize endpoint, arange priorityze increment

def IMS_to_mongo(database: str,collection: str,filePath: str,n_of_test: str,sensors: str,URI='mongodb://localhost:27017',printout=True):
    '''
    ### author: Ariel Priarone - ariel.priarone@studenti.polito.it
    ### Description
    This function take a textual file in the format of the ims dataset and write it to mongoDB for further use.
    Reference: The  data  was  generated  by  the  NSF  I/UCR  Center  for  Intelligent  Maintenance  Systems  
    (IMS  – www.imscenter.net) with support from Rexnord Corp. in Milwaukee, WI. 
    ### Parameters
    ### #database:   
    name of the database to write to.
    #### collection: 
    collection to write to.
    #### filePath:   
    colplete path includeing filename 
    #### n_of_test:  
    number of ims test (1st,2nd,3rd datasets hasve different format)
    #### sensors:    
    list of sensor names to save to mongoDB, as CORRELATED vairables, to save variables as uncorrelated, run this function multiple times for each sensor
    axcepted values:\n'Bearing 1 x'\n'Bearing 1 y'\n'Bearing 2 x'\n'Bearing 2 y'\n'Bearing 3 x'\n'Bearing 3 y'\n'Bearing 4 x'\n'Bearing 4 y'\n'Bearing 1'\n 'Bearing 2'\n'Bearing 3'\n'Bearing 4
    #### URI:        
    URI of mongodb connection. default: 'mongodb://localhost:27017'
    #### print:      
    set to false to suppress print to command line. default=True
    ### Return
    None
    '''
    try:
        type(filePath) == str and type(URI) == str and type(database) == str and type(collection) == str
    except:
        raise Exception("'filePath', 'URI', 'database', 'collection', must all be a string")
    try:
        n_of_test in [1,2,3]
    except:
        raise Exception("'n_of_test' must be 1,2 or 3")
    try:
        all(__x in ["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y","Bearing 1 ", "Bearing 2", "Bearing 3", "Bearing 4"]
            for __x in sensors)
    except:
        raise Exception("'sensors' must be a list subset of: \n 'Bearing 1 x'\n 'Bearing 1 y'\n 'Bearing 2 x'\n 'Bearing 2 y'\n'Bearing 3 x'\n 'Bearing 3 y'\n 'Bearing 4 x'\n 'Bearing 4 y'\n'Bearing 1' \n 'Bearing 2'\n 'Bearing 3'\n 'Bearing 4' ")
    
    match n_of_test:  # names formatting for IMS files
        case 1:
            __names=["Bearing 1 x", "Bearing 1 y", "Bearing 2 x", "Bearing 2 y","Bearing 3 x", "Bearing 3 y", "Bearing 4 x", "Bearing 4 y"]
        case 2:
            __names=["Bearing 1 ", "Bearing 2", "Bearing 3", "Bearing 4"]
        case 3:
            __names=["Bearing 1 ", "Bearing 2", "Bearing 3", "Bearing 4"]
            
    __data=pd.read_csv(filePath,delimiter='\t',names=__names)
    for __i in __data.columns.values.tolist():
        if __i in sensors:
            pass
        else: # remove the unwanted culumns
            __data.drop(labels=__i,axis=1, inplace=True) # axis 1  are columns
    
    # connect to MongoDb
    __client = MongoClient(URI)
    __db = __client[database]
    __collection=__db[collection]
    if printout: 
        print(__name__ +' succesfully connected to the collection in MongoDB')

    # get the timestamp from filename
    __sampleToAadd={'timestamp': IMS_filepathToTimestamp(filePath)} #initialize the dictionary with timestamp
    for __varname in sensors:
        __update={
        __varname:
        {
        'sampFreq': 20000, # typical of IMS files
        'timeSerie': __data[__varname].tolist()
        }    
        }
        __sampleToAadd.update(__update)
    __collection.insert_one(__sampleToAadd)
    if printout: 
        print('\n' + filePath + ' inserted in ' + database + ' ' + collection)


def IMS_filepathToTimestamp(filepath=str):
    __splitted=filepath.split('\\')
    __splitted=__splitted[-1].split('.')
    __int=[int(__splitted[__i]) for __i in range(0,len(__splitted))] # converted in integer values
    return(datetime.datetime(*__int,tzinfo=None))



if __name__=='__main__': 
    pass
    # just for testin, not useful as package functionality
    # print(f'the script \"{os.path.basename(__file__)}\" is running as main!')
    # dirPath = "./data/raw/1st_test_IMSBearing/"
    # fileName = "2003.10.22.12.06.24"
    # dummy=snapshot()
    # dummy.readImsFile(path=dirPath+fileName,variables=["Bearing 1 x", "Bearing 1 y"])
    # print(f"the dataframe has {len(dummy.rawData.index)} rows")
    # print(dummy.rawData.head)
    # print(IMS_filepathToTimestamp(filepath=r'C:\Users\ariel\Documents\Courses\Tesi\Code\data\raw\1st_test_IMSBearing\2003.11.23.20.21.24'))

