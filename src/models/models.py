import copy
from os import error
import pymongo
import src
import os
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
        self.__error_queue_size = self.Config['kmeans']['error_queue_size']
        self.__error_plot_size = self.Config['kmeans']['error_plot_size']
        if self.__error_queue_size > self.__error_plot_size:
            raise ValueError('Error queue size cannot be bigger than the error plot size in "config.yaml"')
        self.novelty_threshold = self.Config['kmeans']['novelty_threshold']
        self.err_dict = {} # dictionary of the error
        self.err_dict['values'] = [0] *self.__error_plot_size # initialize the error array
        self.err_dict['timestamp'] = [0] *self.__error_plot_size # initialize the timestamp array
        self.err_dict['assigned_cluster'] = [0] *self.__error_plot_size # initialize the assigned cluster array
        self.err_dict['anomaly'] = [False] *self.__error_plot_size # initialize the anomaly array
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
            self.__retrieve_StdScaler() # retrieve the scaler
        except:
            self.StdScaler: Dict[str, StandardScaler] = {} # if the scaler is not found, initialize it
        try:
            self.retrieve_KMeans() # retrieve the Kmeans model
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
            os.system('clear')
            match self.mode:
                case 'evaluate':
                    self.evaluate()
                case 'train':
                    if typer.confirm(f"The training procedure will take all the data from the collection '{self.col_features.full_name}' and pack it in the collection '{self.col_train.full_name}'. This will also ERASE the current training data, do you want to PROCEED?", abort=True):
                        self.col_train.delete_many({})
                    while not self.prepare_train_data():
                        pass    # wait for data to be available
                    self.train()
                    if typer.confirm("Do you want to change the 'mode' to 'evaluate'", abort=True):
                        self.mode = 'evaluate'
                case 'retrain':
                    self.retrain()
                case _:
                    self.mode = typer.prompt('Please select the mode of the MLA. The options are: "evaluate", "train" or "retrain"')

    def evaluate(self):
        self.retrieve_KMeans()      # retrieve the Kmeans model
        self.num_clusters = self.kmeans.get_params()['n_clusters'] # get the number of clusters
        self.packFeaturesMatrix()      # pack the training features in a matrix
        while True:
            os.system('clear')
            self.calculate_train_cluster_dist() # calculate the maximum distance of each cluster in the train dataset
            evaluate=False
            while not evaluate: # read the features from the collection
                try:
                    self.snap=self.col_unconsumed.find({'evaluated': {"$exists": False}}).sort('timestamp',pymongo.ASCENDING).limit(1)[0] # get the oldest not evaluated snap
                    evaluate=True
                except IndexError:
                    print(f"No data to evaluate in the '{self.col_unconsumed.full_name}' collection, waiting for new data...")
            self.scale_features()
            if self.evaluate_error():      # evaluate the error - if novelty detected, move to quarantine
                # if the error is positive, move to quarantine
                self._find_snap(self.snap["_id"],self.col_unconsumed) # find the snap in the features collection (to preserve unscaled version)
                self._write_snap(self.col_quarantined) # move the snap to the quarantine collection
            self._mark_snap_evaluated() # mark the snap as evaluated
            self._delete_evaluated_snap() # delete the snap from the unconsumed collection
            print(f"Distance Novelty: {self.err_dict['values'][-1]}")
    
    def _mark_snap_evaluated(self): # to leave at least one snap in the collection for plotting reasons
        self._find_snap(self.snap["_id"],self.col_unconsumed) # find the snap in the features collection (to preserve unscaled version)
        self.snap['evaluated'] = True
        self._replace_snap(self.col_unconsumed) # mark the snap as evaluated
        print(f"Snap '{self.snap['_id']}' marked as evaluated in the '{self.col_unconsumed.full_name}' collection")
    
    def _delete_evaluated_snap(self): # to leave at least one snap in the collection for plotting reasons
        while self.col_unconsumed.count_documents({'evaluated':True}) > 1: # while there are more than one snap to delete
            snap_to_delete = self.col_unconsumed.find({'evaluated':True}).sort('timestamp',pymongo.ASCENDING).limit(1)[0] # get the oldest snap to delete
            self.col_unconsumed.delete_one({'_id': snap_to_delete['_id']}) # delete the snap from the collection
            print(f"Snap '{snap_to_delete['_id']}' deleted from the '{self.col_unconsumed.full_name}' collection")

    def scale_features(self):
        for sensor in self.sensors:
            _data_to_scale = np.array(list(self.snap[sensor].values())).transpose().reshape(1,-1)
            _data_scaled = self.StdScaler[sensor].transform(_data_to_scale).tolist()[0]
            self.snap[sensor] = dict(zip(list(self.snap[sensor].keys()), _data_scaled))

    def evaluate_error(self):
        _features_values = []
        for sensor in self.sensors:
            _features_values.append([float(value) for value in self.snap[sensor].values()])
        _features_values_flat = [item for sublist in _features_values for item in sublist] # flatten the list
        y=self.kmeans.predict(np.array(_features_values_flat).reshape(1,-1)) # predict the cluster for the new snap
        distance_to_assigned_center = self.kmeans.transform(np.array(_features_values_flat).reshape(1,-1))[0,y]

        # the actual estimator of the error is the relative distance margin to the assigned cluster
        current_error=float((distance_to_assigned_center-self.train_cluster_dist[int(y)])/self.train_cluster_dist[int(y)]) # calculate the error
        if self.type == 'fault':
            current_error = -1/current_error # if the type is fault, the error is negative

        anomaly = self.err_dict['values'][-1] > self.novelty_threshold # check if the error is above the threshold
        
        self.err_dict['values'].append(current_error) # append the new error to the error array
        self.err_dict['values'] = self.err_dict['values'][1:] # remove the oldest error from the error array
        self.err_dict['timestamp'].append(self.snap['timestamp']) # append the new error to the timestamp array
        self.err_dict['timestamp'] = self.err_dict['timestamp'][1:] # remove the oldest error from the  timestamp array
        self.err_dict['assigned_cluster'].append(int(y)) # append the new error to the assigned_cluster array
        self.err_dict['assigned_cluster'] = self.err_dict['assigned_cluster'][1:] # remove the oldest error from the  assigned_cluster array
        self.err_dict['anomaly'].append(anomaly) # append the new error to the assigned_cluster array
        self.err_dict['anomaly'] = self.err_dict['assigned_cluster'][1:] # remove the oldest error from the  assigned_cluster array
        
        print(f"Relative distance margin to the assigned cluster #{y}: {self.err_dict['values'][-1]}")

        # write the error to the database
        self.col_models.replace_one({'_id': f'Kmeand cluster {self.type} indicator'}, self.err_dict, upsert=True)

        if anomaly:
            print("NOVELTY DETECTED")
            return True
        else:
            return False

    def calculate_train_cluster_dist(self):
        ''' This method computes the maximum distance of each cluster in the train dataset '''
        self.labels_train_data  = self.kmeans.predict(self.trainMatrix) # predict the cluster of each sample in the train dataset
        self.cluster_distances  = self.kmeans.transform(self.trainMatrix) # gives the distance of each sample to each cluster
        self.train_cluster_dist=[] # maximum distance to eah cluster in the train dataset
        for cluster in range(0,self.num_clusters):
            self.train_cluster_dist.append(max(self.cluster_distances[self.labels_train_data==cluster,cluster])) # get the maximum distance of the samples in the train dataset to tjis cluster
        

    def prepare_train_data(self):
        ''' This method prepares the training data for the MLA '''
        if not self.pack_train_data(): # if the healthy/faulty set is empty, nothing to update
            return False
        self.standardize_features()
        self.save_features_limits()
        self.save_StdScaler()
        return True

    def pack_train_data(self):
        __train_data = self.col_train.find_one({'_id': 'training set'})
        if __train_data is None:  # if the training set is empty, initialize it with the oldest snapshot
            try:
                self.snap = self.col_features.find().sort('timestamp',pymongo.ASCENDING).limit(1)[0]  # get the oldest snapshot
            except IndexError:
                self.__move_to_train()                                          # empty, ask to move all data from unconsumed to train dataset
            __id_to_remove = copy.deepcopy(self.snap['_id'])                  # copy the id of the snapshot to remove
            self.snap['_id']='training set'                                                      # rename it for initializing the training set
            self.col_train.insert_one(self.snap)                                                  # insert it in the training set   
            self.col_features.delete_one({'_id': __id_to_remove})                  # delete the snapshot from the features collection
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
            if self.col_unconsumed.count_documents({}) == 0:
                raise Exception("No data in the collection, cannot initialize the training set")
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
        _train_data_scaled = copy.deepcopy(__train_data)                                   # copy the training set
        _train_data_scaled['_id'] = 'training set scaled'                        # rename it
        self.features_minmax = copy.deepcopy(__train_data)                                   # copy the training set
        self.features_minmax['_id'] = 'training set MIN/MAX'                        # rename it
        
        print(id(__train_data), id(_train_data_scaled), id(self.features_minmax))
        
        # scale the features
        for sensor in self.sensors:
            self.StdScaler[sensor] = StandardScaler()
            __data = np.array(list(__train_data[sensor].values()))      # the scaler wants the data in the form (n_samples, n_features)
            self.StdScaler[sensor].fit(__data.transpose())                              # fit the scaler
            data_scaled = self.StdScaler[sensor].transform(__data.transpose()).transpose()         # the scaler returns the data in the form (n_features, n_samples)
            data_scaled = data_scaled.tolist()                                  # convert the data to list    
            for indx, feature in enumerate(_train_data_scaled[sensor].keys()):
                _train_data_scaled[sensor][feature] = data_scaled[indx]         # the scaler returns the data in the form (n_features, n_samples)
                self.features_minmax[sensor][feature] = [float(np.min(data_scaled[indx])), float(np.max(data_scaled[indx]))]
        # save the scaled data
        self.col_train.delete_many({"_id": 'training set scaled'}) 
        self.col_train.insert_one(_train_data_scaled) 
        print(f"Training set scaled and saved into the collection '{self.col_train.full_name}' with '_id': 'training set scaled'")

    def save_features_limits(self):
        self.col_train.delete_many({"_id": 'training set MIN/MAX'}) 
        self.col_train.insert_one(self.features_minmax)

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
    
    def save_KMeans(self):
        # save the scaler
        __pickled_data = pickle.dumps(self.kmeans)
        __id ='KMeans_'+str(self.type)+'_pickled'
        try:
            self.col_models.insert_one({'_id': __id, 'data': __pickled_data})
        except:
            try:
                self.col_train.replace_one({'_id': __id}, {'_id': __id, 'data': __pickled_data})
            except:
                raise Exception('Error saving the KMeans model')
        print(f"KMeans model saved as picled data into '{self.col_models.full_name}' with '_id': {__id}")
    
    def retrieve_KMeans(self):
        __id ='KMeans_'+str(self.type)+'_pickled'
        __retrieved_data: Collection | None = self.col_models.find_one({'_id': __id})
        if __retrieved_data is None:
            raise Exception('KMeans model not found in collection ' + self.col_models.full_name + ' with _id: ' + __id)
        else:
            self.kmeans:KMeans = pickle.loads(__retrieved_data['data'])
            print(f"KMeans retrieved from picled data @ {self.col_models.full_name}")
    
    def __retrieve_StdScaler(self):
        __retrieved_data: Collection | None = self.col_train.find_one({'_id': 'StandardScaler_pickled'})
        if __retrieved_data is None:
            raise Exception('Scaler not found in collection ' + self.col_train.full_name)
        else:
            self.StdScaler: Dict[str, StandardScaler] = pickle.loads(__retrieved_data['data'])
            print(f"StdScaler retrieved from picled data @ {self.col_train.full_name}")

    @staticmethod
    def retrieve_StdScaler(col: Collection):
        __retrieved_data: Collection | None = col.find_one({'_id': 'StandardScaler_pickled'})
        scaler : Dict[str, StandardScaler] | None
        if __retrieved_data is None:
            scaler = None
        else:
            scaler = pickle.loads(__retrieved_data['data'])
            return scaler

    def _append_features(self, col: Collection):
        ''' Append the features to the collection collection '''
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
        self.save_KMeans()

    
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
        print("Features packed in a matrix: " + str(self.trainMatrix.shape))

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
    NoveltyAgent.evaluate()

    

