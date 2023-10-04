class MLA:
    '''
    Machine Learning Agent:
        - state: str    admitted: 'evaluate', 'train', 'retrain'
    '''
    def __init__(self, state: str):
        self.__state = state

    @property
    def state(self):
        return self.__state
    
    @state.setter
    def state(self, value: str):
        if value not in ['evaluate', 'train', 'retrain']:
            raise ValueError('Invalid state')
        else:
            self.__state = value
