import pymongo
import src
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
import numpy as np
from rich import print
import pickle
from typing import List, Dict, Tuple
import typer


class MLA(src.data.DB_Manager):
    '''
    Machine Learning Agent:
    '''
    def __init__(self, configStr: str, type: str = 'novelty'):
        super().__init__(configStr)
        self.type = type              #  type of the MLA (novelty/fault) - how normal/how faulty the data are
        self.__mode = self.Config['MLA']['mode']
        match self.type:
            case 'novelty':
                self.col_features = self.col_healthy
                self.col_train = self.col_healthy_train
            case 'fault':
                self.col_features = self.col_faulty
                self.col_train = self.col_faulty_train
            case _:
                raise ValueError('Type of MLA is not valid. It should be either "novelty" or "fault", but it is: ' + self.type)
        try:
            self.retrieve_StdScaler() # retrieve the scaler
        except:
            self.StdScaler: Dict[str, StandardScaler] = {} # if the scaler is not found, initialize it

    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, value: str):
        if value not in ['evaluate', 'train', 'retrain']:
            raise ValueError('Invalid state')
        else:
            self.__mode = value

    def run(self):
        '''Run the MLA according to its state'''
        match self.mode:
            case 'evaluate':
                self.evaluate()
            case 'train':
                self.prepare_train_data()
                self.train()
            case 'retrain':
                self.retrain()

    def evaluate(self):
        pass

    def prepare_train_data(self):
        ''' This method prepares the training data for the MLA '''
        self.pack_train_data()
        self.standardize_features()
        self.save_StdScaler()

    def pack_train_data(self):
        __train_data = self.col_train.find_one({'_id': 'training set'})
        if __train_data is None:  # if the training set is empty, initialize it with the oldest snapshot
            try:
                self.snap = self.col_features.find().sort('timestamp',pymongo.ASCENDING).limit(1)[0]  # get the oldest snapshot
            except IndexError:
                print(f"No data in the '{self.col_features.full_name}' collection, waiting for new data...")
                if typer.confirm(f"Do you want to move all data from {self.col_unconsumed.name} to {self.col_features.name}?"):
                    self.moveCollection(self.col_unconsumed, self.col_features)
                else:
                    print("Exiting...")
                    raise Exception("No data in the collection, cannot initialize the training set")
            self.snap['_id']='training set'                                                      # rename it for initializing the training set
            self.col_train.insert_one(self.snap)                                                  # insert it in the training set   
            print("Training set initialized") 
        else:                   # append healty documents to the dataset
            cursor = self.col_features.find().sort('timestamp',pymongo.ASCENDING)  # get the oldest snapshot
            for self.snap in cursor:
                if isinstance(__train_data['timestamp'],list):                     # if the training set is a list, pass
                    pass
                else:                                                            # convert everityng to list
                    __train_data['timestamp'] = [__train_data['timestamp']]
                    for sensor in self.sensors:
                        for feature in __train_data[sensor].keys():
                            __train_data[sensor][feature] = [__train_data[sensor][feature]]          
                __train_data['timestamp'].append(self.snap['timestamp'])                  # append the timestamp
                for sensor in self.sensors:
                    for feature in __train_data[sensor].keys():
                        __train_data[sensor][feature].append(self.snap[sensor][feature])  # append the sensor data
                self.col_features.delete_one({'_id': self.snap['_id']})                  # delete the snapshot from the features collection
        
            self.col_train.replace_one({'_id': 'training set'}, __train_data)         # replace the training set with the updated one 
            print("Training set updated")

    def standardize_features(self):
        # now this method scales the data
        __train_data = self.col_train.find_one({'_id': 'training set'})             # get the training set
        if __train_data is None:
            raise Exception('Training set not initialized')
        __train_data_scaled = __train_data.copy()                                   # copy the training set
        __train_data_scaled['_id'] = 'training set scaled'                        # rename it
        # scale the features
        for sensor in self.sensors:
            self.StdScaler[sensor] = StandardScaler()
            __data = np.array(list(__train_data[sensor].values()))      # the scaler wants the data in the form (n_samples, n_features)
            self.StdScaler[sensor].fit(__data.transpose())                              # fit the scaler
            data_scaled = self.StdScaler[sensor].transform(__data.transpose()).transpose()         # the scaler returns the data in the form (n_features, n_samples)
            data_scaled = data_scaled.tolist()                                  # convert the data to list    

            for indx, feature in enumerate(__train_data_scaled[sensor].keys()):
                __train_data_scaled[sensor][feature] = data_scaled[indx]         # the scaler returns the data in the form (n_features, n_samples)
        # save the scaled data
        self.col_train.delete_many({"_id": 'training set scaled'}) 
        self.col_train.insert_one(__train_data_scaled) 
        print("Training set scaled")
    
    def save_StdScaler(self):
        # save the scaler
        __pickled_data = pickle.dumps(self.StdScaler)
        try:
            self.col_train.insert_one({'_id': 'StandardScaler_pickled', 'data': __pickled_data})
        except:
            try:
                self.col_train.replace_one({'_id': 'StandardScaler_pickled'}, {'_id': 'StandardScaler_pickled', 'data': __pickled_data})
            except:
                raise Exception('Error saving the StandardScaler')
    
    def retrieve_StdScaler(self):
        __retrieved_data: Collection | None = self.col_train.find_one({'_id': 'StandardScaler_pickled'})
        if __retrieved_data is None:
            raise Exception('Scaler not found')
        else:
            self.StdScaler = pickle.loads(__retrieved_data['data'])
            print(f"StdScaler retrieved from picled data @ {__retrieved_data.full_name}")

    def _read_features(self, col: Collection, order = pymongo.ASCENDING):
        ''' Read the data from the collection - put data in self.snap
            return True if data are available, False otherwise '''
        try:
            self.snap    = col.find().sort('timestamp',order).limit(1)[0]     # oldest/newest record - sort gives a cursor, the [0] is the dict
            print(f"Imported snapshot with timestamp {self.snap['timestamp']} from {col}")
            return True    
        except IndexError:
            print(f"No data in collection {col.full_name}, waiting for new data...")
            return False
        
    def _write_features(self, col: Collection):
        ''' Write the data to the collection '''
        col.insert_one(self.snap)
        print(f"Inserted snapshot with timestamp {self.snap['timestamp']} into {col}")

    def _append_features(self, col: Collection):
        ''' Append the features to the train collection '''
        col.update_one({'_id': 'trainig set'}, {'$set': {}})
    

    def train(self):
        pass

    def retrain(self):
        pass

    

