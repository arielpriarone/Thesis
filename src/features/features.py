from ast import Dict

from matplotlib.markers import MarkerStyle
import src
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy
from rich import print
import pymongo
import matplotlib.cm as cm
from itertools import chain
from matplotlib.lines import Line2D
from typing import Dict, List, Union, Optional, Tuple

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
        self.__last_snap_timestamp = None                                               # timestamp of the last snapshot plotted
    
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

    def initialize_barPlotFeatures(self,axs: plt.Axes):
        try:
            snap = self.col_unconsumed.find().sort('timestamp', pymongo.DESCENDING).limit(1)[0]  # latest document in collection
        except IndexError:
            print('No data in collection, wait for new data...')
            return None
        try:
            self.MinMax = self.col_healthy_train.find({'_id': 'training set MIN/MAX'})[0]
        except IndexError:
            self.MinMax = None
        tab10_cmap = cm.get_cmap("Set1")
        self.__colors = [tab10_cmap(indx)[:3] for indx, _ in enumerate(self.sensors)] # convert tuple to list
        self.__base_width = 1.0     # the width of the bars
        self.__separator  = 0.5   # the space between the bars
        self.__features_list = []  # list of all features
        self.__Scaler = src.models.MLA.retrieve_StdScaler(col=self.col_healthy_train)

        for sensor in self.sensors:
            self.__features_list.append(list(snap[sensor].keys()))
        self.features_list = list(chain.from_iterable(self.__features_list))  # flatten list
        self.features_list = list(dict.fromkeys(self.features_list))  # remove duplicates
        self.__feature_mask = {key: [False] * len(self.sensors) for key in self.features_list} # initialize dictionary
        for sensor_number, sensor in enumerate(self.sensors):
            for feature in self.features_list:
                if feature in snap[sensor].keys():
                    self.__feature_mask[feature][sensor_number] = True
        self.__locator_bars = [0.0]  # the x locations for the groups
        self.__locator_ticks = []  # the x locations for the ticks
        self.__minMax = [0.0,0.0]
        for feature in self.features_list:        
            for sensor_number, sensor in enumerate(self.sensors):
                if self.__feature_mask[feature][sensor_number] and  self.MinMax is not None:                        
                    self.__minMax[0] = min(self.__minMax[0], self.MinMax[sensor][feature][0])
                    self.__minMax[1] = max(self.__minMax[1], self.MinMax[sensor][feature][1])
        
        for feature in self.features_list:
            width = self.__base_width
            offset = 0.0          
            for sensor_number, sensor in enumerate(self.sensors):
                if self.__feature_mask[feature][sensor_number]:                        
                    offset += width
            self.__locator_ticks.append(self.__locator_bars[-1] + (offset-width) / 2 if offset > 0 else self.__locator_bars[-1])
            self.__locator_bars.append(self.__locator_bars[-1] + offset + self.__separator)

        self.__legend_lines = [Line2D([0], [0], color=self.__colors[indx], lw=4, label=sensor) for indx, sensor in enumerate(self.sensors)] # type: ignore
        self.__legend_labels = copy.deepcopy(self.sensors)
        if self.MinMax is not None:
            self.__legend_lines.extend([Line2D([0], [0], color=self.__colors[indx], lw=4, alpha=1) for indx, sensor in enumerate(self.sensors)]) # type: ignore
            minmax_legend = [f"{sensor} min/max record" for sensor in self.sensors]
            self.__legend_labels.extend(minmax_legend)
            axs.set_ylim(ymin=self.__minMax[0]*1.1, ymax=self.__minMax[1]*1.3) # type: ignore
        
        return axs

    def barPlotFeatures(self,axs: plt.Axes):
        """
        Plots a bar chart of the latest features for each sensor in the collection.

        Parameters:
        axs (matplotlib.axes.Axes): The axes on which to plot the bar chart.

        Returns:
        matplotlib.axes.Axes: The axes on which the bar chart was plotted.
        """
        try:
            snap = self.col_unconsumed.find().sort('timestamp', pymongo.DESCENDING).limit(1)[0]  # latest document in collection
        except IndexError:
            print('No data in collection, wait for new data...')
            return
        if snap['timestamp'] == self.__last_snap_timestamp:
            print('Latest data already plotted... waiting for new data...')
            return
        axs.clear()  # Clear last data frame
        axs.set_title(f"Latest features for each sensor. Timestamp: {snap['timestamp']}")  # set title
        if self.__Scaler is not None:      # if the scaler is available, scale the data
            for sensor in self.sensors:
                _data_to_scale = np.array(list(snap[sensor].values())).transpose().reshape(1,-1)
                _data_scaled = self.__Scaler[sensor].transform(_data_to_scale).transpose().tolist()
                snap[sensor] = dict(zip(list(snap[sensor].keys()), _data_scaled))
                axs.set_title(f"Latest standardized features for each sensor. Timestamp: {snap['timestamp']}")
    
        for bar_group, feature in enumerate(self.features_list):
            width = self.__base_width
            offset = 0.0          
            for sensor_number, sensor in enumerate(self.sensors):
                if self.__feature_mask[feature][sensor_number]:                        
                    axs.bar(self.__locator_bars[bar_group]+offset, snap[sensor][feature], width, color=self.__colors[sensor_number], alpha=1)
                    if self.MinMax is not None:
                        axs.scatter(self.__locator_bars[bar_group]+offset, self.MinMax[sensor][feature][0], marker=6, s=15, color=self.__colors[sensor_number], alpha=1) # type: ignore
                        axs.scatter(self.__locator_bars[bar_group]+offset, self.MinMax[sensor][feature][1], marker=7, s=15, color=self.__colors[sensor_number], alpha=1) # type: ignore
                    offset += width

        axs.set_xticks(self.__locator_ticks,self.features_list)
        axs.tick_params(axis='x',rotation = 90)
        axs.legend(self.__legend_lines, self.__legend_labels, loc='upper right',  ncol=len(self.sensors)*2)
        axs.set_ylabel('Feature value [-]')
        axs.set_xlabel('Features [-]')
        axs.spines['left'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.grid(True,which='both',axis='x')
        self.__last_snap_timestamp = snap['timestamp']
        if __name__=='__main__':
            plt.show()
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
    FeatureAgent=FA(r'C:\Users\ariel\Documents\Courses\Tesi\Code\config.yaml')
    fig, ax = plt.subplots()
    FeatureAgent.initialize_barPlotFeatures(ax)
    FeatureAgent.barPlotFeatures(ax)
    # FeatureAgent.run()


    