import src
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt
import time
import multiprocessing
from rich import print
import pymongo
import matplotlib.cm as cm

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
        try:
            self.snap    = self.col_raw.find().sort('timestamp',self.order).limit(1)[0]     # oldest/newest record - sort gives a cursor, the [0] is the dict
            print(f"Imported snapshot with timestamp {self.snap['timestamp']} from {self.col_raw}")
            return True    
        except IndexError:
            print(f"No data in collection {self.col_raw.full_name}, waiting for new data...")
            return False

    def _extractFeatures(self):
        ''' extract features from the data '''
        for sensor in self.sensors:                                                         # for each sensor (names are keys of the dict)
            self.features["timestamp"] = self.snap["timestamp"]                             # add the timestamp to the features
            self._extractTimeFeautures(sensor)                                                   # extract time domain features
            self._extractFreqFeautures(sensor)                                                   # extract frequency domain features

    def _extractTimeFeautures(self, sensor):
        ''' extract time domain features '''
        # if Mean Enabled
        if self.Config['Database']['sensors'][sensor]['features']['mean']:
            self.features[sensor].update({'mean':np.mean(self.snap[sensor]['timeSerie'])})
            print(f"Mean extracted from [purple]{sensor}[/]")
        # if RMS Enabled
        if self.Config['Database']['sensors'][sensor]['features']['rms']:
            self.features[sensor].update({'rms':np.sqrt(np.mean(np.square(self.snap[sensor]['timeSerie'])))})
            print(f"RMS extracted from [purple]{sensor}[/]")
        # if peak2peak Enabled
        if self.Config['Database']['sensors'][sensor]['features']['peak']:
            self.features[sensor].update({'peak2peak':np.ptp(self.snap[sensor]['timeSerie'])})
            print(f"Peak2Peak extracted from [purple]{sensor}[/]")
        # if std Enabled
        if self.Config['Database']['sensors'][sensor]['features']['std']:
            self.features[sensor].update({'std':np.std(self.snap[sensor]['timeSerie'])})
            print(f"Standard deviation extracted from [purple]{sensor}[/]")

    def _extractFreqFeautures(self, sensor):
        # if Wavelet is enabled
        if self.Config['Database']['sensors'][sensor]['features']['wavPowers']:    
            _, pows, nodes, _, _ = packTrasform(self.snap[sensor]['timeSerie'],     # perform the wavelet trasform
                                                wavelet=self.Config["wavelet"]["type"],
                                                mode=self.Config["wavelet"]["mode"],
                                                maxlevel=self.Config["wavelet"]["maxlevel"], 
                                                plot=False)
            self.features[sensor].update(dict(zip(nodes, pows)))  # create a dictionary with nodes as keys and powers as values
            print(f"Wavelet coefs extracted from [purple]{sensor}[/]")
                
    def _deleteFromraw(self):
        ''' delete current snap record from the RAW collection '''
        self.col_raw.delete_one({'_id':self.snap['_id']})
        print(f"Deleted snapshot with timestamp {self.snap['timestamp']} from {self.col_raw}")

    def _writeToUnconsumed(self):
        ''' write the extracted features to the Unconsumed collection '''
        __dummy=self.features.copy() # create a copy of the features dictionary
        self.col_unconsumed.insert_one(__dummy) # insert the features in the Unconsumed collection, without changing the dictionary

    def barPlotFeatures(self, axs: plt.Axes):
        try:
            snap = self.col_unconsumed.find().sort('timestamp', pymongo.DESCENDING).limit(1)[0]  # latest document in collection
        except IndexError:
            print('No data in collection, wait for new data...')
            return
        
        tab10_cmap = cm.get_cmap("tab10")
        ticks = []; 
        bars = []
        colors = []
        labels = []
        for sens in self.sensors:
            for keys in snap[sens].keys():
                ticks.append(keys)
                bars.append(snap[sens][keys])
                colors.append(tab10_cmap(self.sensors.index(sens)))
                labels.append(sens)
        axs.clear()  # Clear last data frame
        axs.bar(ticks,bars,color=colors,label=labels)  # Plot new data frame
        axs.legend()
        return axs

    def run(self):
        while True:
            while not self._readFromRaw(): pass  # wait for new data
            self._extractFeatures()
            self._writeToUnconsumed()
            self._deleteFromraw()
    
    

if __name__=='__main__': 
    # just for testin, not useful as package functionality
    # timeSerie=src.data.readSnapshot('IMS','RAW','mongodb://localhost:27017')['Bearing 1 x']['timeSerie']
    # coef, pows, nodes, _, _ = packTrasform(timeSerie, plot=True)
    # plt.show()
    FeatureAgent=FA("../config.yaml")
    FeatureAgent.run()

    