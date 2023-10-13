import pymongo
import src
from pymongo.collection import Collection
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples
from rich import print
import pickle
from typing import List, Dict, Tuple
import typer
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt

class MLA(src.data.DB_Manager):
    '''
    Machine Learning Agent:
    '''
    def __init__(self, configStr: str, type: str = 'novelty'):
        super().__init__(configStr)
        self.type = type              #  type of the MLA (novelty/fault) - how normal/how faulty the data are
        self.__max_clusters = self.Config['kmeans']['max_clusters']
        self.__max_iter = self.Config['kmeans']['max_iterations']
        self.__mode: str | None = None              #  mode of the MLA (evaluate/train/retrain)
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
        try:
            self.retrieve_Kmeans() # retrieve the Kmeans model
        except:
            self.kmeans=KMeans()

    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, value: str):
        if value not in ['evaluate', 'train', 'retrain']:
            print('Mode not valid. It should be either "evaluate", "train" or "retrain", but it is: ' + value)
        else:
            self.__mode = value

    def run(self):
        '''Run the MLA according to its state'''
        while True:
            match self.mode:
                case 'evaluate':
                    self.evaluate()
                case 'train':
                    while not self.prepare_train_data():
                        pass    # wait for data to be available
                    self.train()
                case 'retrain':
                    self.retrain()
                case _:
                    self.mode = typer.prompt('Please select the mode of the MLA. The options are: "evaluate", "train" or "retrain"')

    def evaluate(self):
        pass

    def prepare_train_data(self):
        ''' This method prepares the training data for the MLA '''
        if not self.pack_train_data(): # if the healthy/faulty set is empty, nothing to update
            return False
        self.standardize_features()
        self.save_StdScaler()
        return True

    def pack_train_data(self):
        __train_data = self.col_train.find_one({'_id': 'training set'})
        if __train_data is None:  # if the training set is empty, initialize it with the oldest snapshot
            try:
                self.snap = self.col_features.find().sort('timestamp',pymongo.ASCENDING).limit(1)[0]  # get the oldest snapshot
            except IndexError:
                self.__move_to_train()                                          # empty, ask to move all data from unconsumed to train dataset
            self.snap['_id']='training set'                                                      # rename it for initializing the training set
            self.col_train.insert_one(self.snap)                                                  # insert it in the training set   
            print("Training set initialized to '{self.col_train.full_name}' with '_id': 'training set'") 
        else:                   # append healty documents to the dataset
            if self.col_features.count_documents({}) == 0:
                self.__move_to_train()                                        # empty, ask to move all data from unconsumed to train dataset
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
            print(f"Training set updated to '{self.col_train.full_name}' with '_id': 'training set' ")
            return True # return True if the training set has been updated

    def __move_to_train(self):
        print(f"No data in the '{self.col_features.full_name}' collection, waiting for new data...")
        if typer.confirm(f"Do you want to move 'ALL' data  from '{self.col_unconsumed.full_name}' to '{self.col_features.full_name}'?",default=False):
            self.moveCollection(self.col_unconsumed, self.col_features)
            self.snap = self.col_features.find().sort('timestamp',pymongo.ASCENDING).limit(1)[0]  # get the oldest snapshot
        else:
            print("Exiting...")
            raise Exception("No data in the collection, cannot initialize the training set")

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
        print(f"Training set scaled and saved into the collection '{self.col_train.full_name}' with '_id': 'training set scaled'")
    
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
        print(f"StandardScaler saved as picled data into '{self.col_train.full_name}' with '_id': 'StandardScaler_pickled'")
    
    def retrieve_StdScaler(self):
        __retrieved_data: Collection | None = self.col_train.find_one({'_id': 'StandardScaler_pickled'})
        if __retrieved_data is None:
            raise Exception('Scaler not found in collection ' + self.col_train.full_name)
        else:
            self.StdScaler = pickle.loads(__retrieved_data['data'])
            print(f"StdScaler retrieved from picled data @ {__retrieved_data.full_name}")

    def retrieve_Kmeans(self):
        __retrieved_data: Collection | None = self.col_models.find_one({'_id': 'Kmeans_pickled'})
        if __retrieved_data is None:
            raise Exception('Kmeans not found in collection ' + self.col_models.full_name)
        else:
            self.kmeans = pickle.loads(__retrieved_data['data'])
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
        self.packFeaturesMatrix()       # pack the training features in a matrix
        self.evaluate_silhouette()
        self.evaluate_inertia()

        fig, axs=plt.subplots(1,2)
        fig.tight_layout()
        self.__plot_silhouette(axs[0])
        self.__plot_inertia(axs[1])
        print("Please decide the number of cluster for the training. The silhouette and inertia plots will be shown.")
        print("The silhouette should be maximized, while the inertia should be in a Pareto optimal point.")
        print("close the plot to continue...")
        plt.show()
        self.num_clusters=typer.prompt("Number of clusters", type=int)

        self.kmeans=KMeans(self.num_clusters,n_init='auto',max_iter=self.__max_iter) #reinitialize the kmeans
        self.kmeans.fit(self.trainMatrix)
        print("Kmeans trained with " + str(self.num_clusters) + " clusters")

    
    def evaluate_silhouette(self):
        ''' This method evaluates the silhouette score for the training set '''
        self.__sil_score=[]
        for n_blobs in range(2,self.__max_clusters+1):
            __kmeans=KMeans(n_blobs,n_init='auto',max_iter=self.__max_iter)
            __y_pred_train=__kmeans.fit_predict(self.trainMatrix )
            self.__sil_score.append(silhouette_score(self.trainMatrix,__y_pred_train))
    
    def evaluate_inertia(self):
        self.__inertia=[]
        for n_blobs in range(1,self.__max_clusters+1):
            kmeans=KMeans(n_blobs,n_init='auto',max_iter=1000)
            kmeans.fit_predict(self.trainMatrix)
            self.__inertia.append(kmeans.inertia_)

    def packFeaturesMatrix(self):
        ''' This method packs the training features in a matrix'''
        __train_data = self.col_train.find_one({'_id': 'training set scaled'})             # get the training set
        if __train_data is None:
            raise Exception("'_id': 'training set scaled' not found in collection " + self.col_train.full_name)
        __features_values = []
        __features_names = []
        for sensor in self.sensors:
            for feature in __train_data[sensor].keys():
                __features_names.append(sensor + '_' + feature)
                __features_values.append(__train_data[sensor][feature])
        self.features_names, self.trainMatrix = (__features_names, np.array(__features_values).transpose())

    def __plot_silhouette(self, ax):
        ax.plot(range(2,self.__max_clusters+1),self.__sil_score)
        ax.set_ylabel('Silhouette')
        ax.set_xlabel('Num. of clusters')

    def __plot_inertia(self, ax):
        ax.plot(range(1,self.__max_clusters+1),self.__inertia)
        ax.set_ylabel('Inertia')
        ax.set_xlabel('Num. of clusters')

    def retrain(self):
        pass

if __name__ == '__main__':
    NoveltyAgent = MLA(configStr=r"C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml", type='novelty')
    NoveltyAgent.run()

    

