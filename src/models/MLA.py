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
                self.train()
            case 'retrain':
                self.retrain()

    def evaluate(self):
        pass

    def train(self):
        '''
        Steps of the functions
            - pick new samples from the HEALTY/FAULTY database
            - move it to the TRAIN database
            - continue untill all are moved
            - train the model
        '''
        # pick new samples from the HEALTY/FAULTY database
        match self.type:
            case 'novelty':
                self._read_features(self.col_healthy, pymongo.ASCENDING)
            case 'fault':
                self._read_features(self.col_faulty, pymongo.DESCENDING)
            case _:
                raise ValueError('Type of MLA is not valid. It should be either "novelty" or "fault", but it is: ' + self.type)
        
    def _read_features(self, col: Collection, order = pymongo.ASCENDING):
        ''' Read the data from the collection '''
        try:
            self.snap    = col.find().sort('timestamp',order).limit(1)[0]     # oldest/newest record - sort gives a cursor, the [0] is the dict
            print(f"Imported snapshot with timestamp {self.snap['timestamp']} from {col}")
            return True    
        except IndexError:
            print(f"No data in collection {col.full_name}, waiting for new data...")
            return False
    

    def retrain(self):
        pass

    

