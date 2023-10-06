import src
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt
import time
import multiprocessing
from rich import print

def FFT(array,samplFreq=1,preproc=None):
    # this function perform the FFT trasform of a signal with windowind preprocessing
    # it return the FFT array (freq domain), the frequency points (freq domain)
    # and the windowed array (time domaint) preprocessing the data
    array=np.asarray(array)
    match preproc:
        case 'Hann':
            window = np.hanning(len(array))
            _prepArray=np.multiply(array,window) #preprocessed array
        case 'Hamming':
            window = np.hamming(len(array))
            _prepArray=np.multiply(array,window) #preprocessed array
        case 'Flip':
            np.flip(array[0:len(array)-1])
            _prepArray=np.concatenate((array, np.flip(array[0:len(array)-1])), axis=0)
        case _:
            _prepArray=array
    _aux = np.fft.fft(_prepArray)/len(_prepArray)          # Normalize amplitude
    if preproc=='Flip':
        _aux=_aux[::2] # if flip the number of point doubled, now i drop the odd numbers
    _timePeriod  = len(_prepArray)/samplFreq
    _frequencies = np.arange(int(len(_prepArray)/2))/_timePeriod #frequencies array of fft
    return abs(_aux[range(int(len(_prepArray)/2))]), _frequencies, _prepArray

def packTrasform(timeSerie: list,wavelet='db10', mode='symmetric',maxlevel=6, plot=False):
    '''perform the wavelet trasform of a time series:
    RETURN: coefs:  [list] coefficients of the decomposition
            pows:   [list] powers of all the coefficients
            nodes:  [list] names of the nodes'''
    _wp = pywt.WaveletPacket(data=timeSerie, wavelet=wavelet, mode=mode,maxlevel=maxlevel)   # perform the packet trasform
    _nodes=[node.path for node in _wp.get_level(_wp.maxlevel, 'natural')]                    # extract the lower level nodes
    _powers=[np.linalg.norm(_wp[index].data) for index in _nodes]                            # compute the l2 norm of coefs
    _coefs=[_wp[index].data for index in _nodes]
    fig, axs   = plt.subplots()
    if plot:
        axs.bar(_nodes, _powers); axs.tick_params(axis='x',rotation = 90)
        axs.set_ylabel('Power [-]')
        axs.set_xlabel('Nodes [-]')
        plt.show()

    return _coefs, _powers, _nodes, fig, axs

class FA(src.data.DB_Manager):
    '''
    empty the RAW collection and populate the Unconsumed collection with extracted features:
    '''
    def __init__(self, configStr: str, order: int = 1):
        super().__init__(configStr)
        if order not in [-1,1]:
            raise ValueError('order must be either latest or oldest')
        self.order  =   order                                                           # pick -1=latest / 1=oldest raw data available
    
    def _readFromRaw(self):
        ''' Read the data from the RAW collection '''
        self.snap    = self.col_raw.find().sort('timestamp',self.order).limit(1)[0]     # oldest/newest record - sort gives a cursor, the [0] is the dict
        sensors      = list(self.snap.keys())[2::]                                      # current sensors names
        if not set(sensors).issubset(set(self.Config['Database']['sensors'])):
            raise ValueError(f'sensors found in the collection {self.col_unconsumed} not in the configuration file')
        self.sensors  = list(self.snap.keys())[2::]                                     # current sensors names      
        print(f"Snapshot with timestamp {self.snap['timestamp']} read from {self.col_raw}")       

    def _extractFeatures(self):
        ''' extract features from the data '''
        pass
    def _deleteFromraw(self):
        ''' delete a record from the RAW collection '''
        pass
    def _writeToUnconsumed(self):
        ''' write the extracted features to the Unconsumed collection '''

    def run(self):
        while True:
            self._readFromRaw()
            self._extractFeatures()
            self._deleteFromraw()
            self._writeToUnconsumed()
    
    

if __name__=='__main__': 
    # just for testin, not useful as package functionality
    # timeSerie=src.data.readSnapshot('IMS','RAW','mongodb://localhost:27017')['Bearing 1 x']['timeSerie']
    # coef, pows, nodes, _, _ = packTrasform(timeSerie, plot=True)
    # plt.show()
    FeatureAgent=FA("../config.yaml")
    print(FeatureAgent.Config)
    FeatureAgent._readFromRaw()

    