import src
import numpy as np
from sklearn.preprocessing import StandardScaler
import pywt
import matplotlib.pyplot as plt

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


def featExtraction():
    pass

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

if __name__=='__main__': 
    # just for testin, not useful as package functionality
    timeSerie=src.data.readSnapshot('IMS','RAW','mongodb://localhost:27017')['Bearing 1 x']['timeSerie']
    coef, pows, nodes, _, _ = packTrasform(timeSerie, plot=True)
    plt.show()
    