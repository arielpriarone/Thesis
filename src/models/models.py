import pymongo
import src
from pymongo.collection import Collection

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

    @property
    def mode(self):
        return self.__mode
    
    @mode.setter
    def mode(self, value: str):
        if value not in ['evaluate', 'train', 'retrain']:
            raise ValueError('Invalid state')
        else:
            self.__state = value

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
        '''''
        Steps of the functions
            - pick the features from the collection
            - append them in the train collection
            - remove from original collection
            - scale the features
        '''
        self.col_train.find_one({'_id': 'training set'})
        # move all documents from feature to TRAIN collection
        self._read_features(self.col_features, pymongo.ASCENDING) # read the oldest record
    
    def _standardize_features(self, col: Collection):
        ''' Standardize the features in the collection '''
        pass
        
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

    

