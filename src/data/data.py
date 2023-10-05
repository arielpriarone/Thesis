import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot  as plt
from pymongo import MongoClient
import json5 as json
from rich import print

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

class DB_Manager:
    '''
    class for managing the whole Database system - upgrade from the loose functions previously used
    '''
    def __init__(self, configStr: str):
        self.configStr = configStr    #  path to config file (json)
        try:
            self.Config = json.load(open(self.configStr))
        except:
            raise Exception(f'Error reading config file @ {self.configStr}')
        self.client, self.db, self.col_back = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['back'],self.Config['Database']['URI'])
        _, _, self.col_raw = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['raw'],self.Config['Database']['URI'])
        _, _, self.col_unconsumed = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['unconsumed'],self.Config['Database']['URI'])
        _, _, self.col_healthy = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['healthy'],self.Config['Database']['URI'])
        _, _, self.col_quarantined = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['quarantined'],self.Config['Database']['URI'])
        _, _, self.col_faulty= mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['faulty'],self.Config['Database']['URI'])
        _, _, self.col_models = mongoConnect(self.Config['Database']['db'],self.Config['Database']['collection']['models'],self.Config['Database']['URI'])
    @staticmethod
    def createEmptyDB(configStr: str):
        '''
        create an empty database with the collections specified in the config file.
        '''
        try:
            Config = json.load(open(configStr))
        except:
            raise Exception(f'Error reading config file @ {configStr}')
        client  = MongoClient(Config['Database']['URI'])                                    # connect to MongoBD
        if Config['Database']['db'] in client.list_database_names():
            raise Exception(f'Database \'{Config["Database"]["db"]}\' already exists @ \'{Config["Database"]["URI"]}\'')
        else:
            db = client[Config['Database']['db']]                                           # create database
            print(f'Created empty database \'{Config["Database"]["db"]}\' @ \'{Config["Database"]["URI"]}\'')
            for cols in Config['Database']['collection'].values():
                db.create_collection(cols)                                                  # create empty collections
                print(f'Created empty collection \'{cols}\' @ \'{Config["Database"]["db"]}\'')
        client.close()                                                                      # close connection

        
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
    return(datetime(*__int,tzinfo=None))

def readSnapshot(database: str,collection: str,URI: str,timestamp='',plot=False):
    '''
    Read the oldest snapshot from mongoDB and provide dictionary with the elements, optional plot available
    INPUT:
        database: str       name of the dateabase
        collection: str     name of the collection
        URI: str            URI of the database
        timestamp=''        if '' the oldest record is collected
        plot=False          plot the data or just return the dictionary
    RETURN: snap - dicttionary of the snapshot
    EXAMPLE: readOldSnapshot('IMS','RAW','mongodb://localhost:27017',timestamp='2003-10-22T12:06:24.000+00:00',plot=True)
    '''
    client, db, col = mongoConnect(database,collection,URI)
    if timestamp == '':
        snap    = col.find().sort('timestamp',1).limit(1)[0]    # oldest record - sort gives a cursor, the [0] is the dict
    else:
        mydate  = datetime.fromisoformat(timestamp)
        snap    = col.find({'timestamp': mydate})[0]            #pick the right snapshot
    _sens  = list(snap.keys())[2::]                             # sensors to iterate
    if plot:
        fig, axs = plt.subplots(len(_sens))
        for _i, _sen in enumerate(_sens):
            _time = np.linspace(0,len(snap[_sen]['timeSerie'])/snap[_sen]['sampFreq'],len(snap[_sen]['timeSerie']))
            axs[_i].plot(_time,snap[_sen]['timeSerie'])
            axs[_i].grid('both');axs[_i].set_ylabel(_sen)
        axs[-1].set_xlabel('Time [s]')
        aux='timestamp'; axs[0].set_title(f'Plot of the snapshot {snap[aux]}')
        plt.show()
    return snap

def mongoConnect(database: str,collection: str,URI: str):
    '''
    connect to a MongoDB server collection, return the client, the database and the collection.
    this is meant to connect to an EXISTING colleciton!
    return: client, database and collection
    '''
    client  = MongoClient(URI)          # connect to MongoBD
    if database in client.list_database_names():
        db      = client[database]          # connect database
    else:
        raise ConnectionError(f'Database \'{database}\' not found @ \'{URI}\'')
    if collection in db.list_collection_names():
        col     = db[collection]            # connect to collection
    else:
        raise ConnectionError(f'Collection \'{collection}\' not found in Database \'{database}\' @ \'{URI}\'')
    return client, db, col


if __name__=='__main__': 
    # just for testin, not useful as package functionality
    #print(readSnapshot('IMS','RAW','mongodb://localhost:27017'))
    #client, db, col = mongoConnect('IMS','RAW','mongodb://localhost:27017')
    #print(type(client),type(db),type(col))
    #DB_Manager= DB_Manager('../config.json')
    DB_Manager.createEmptyDB(r'C:\Users\ariel\Documents\Courses\Tesi\Code\config.json5')