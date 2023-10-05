import pymongo
import src

class MLA:
    '''
    Machine Learning Agent:
        - state: str    admitted: 'evaluate', 'train', 'retrain'
    '''
    def __init__(self, state: str, database: str,collection: str,URI: str):
        self.__state = state
        self.__database = database
        self.__collection = collection
        self.__URI = URI

    @property
    def state(self):
        return self.__state
    
    @state.setter
    def state(self, value: str):
        if value not in ['evaluate', 'train', 'retrain']:
            raise ValueError('Invalid state')
        else:
            self.__state = value

    def run(self):
        '''Run the MLA according to its state'''
        match self.state:
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
        pass

    def retrain(self):
        pass

    

